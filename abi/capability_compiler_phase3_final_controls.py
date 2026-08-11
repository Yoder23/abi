"""Matched causal controls for the final Phase 3 corrective acquisition stage."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_sparse_router as sparse
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
from .capability_compiler_phase3_dual_view_recovery import (
    DualViewSampler,
    dual_examples,
)
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_host_recovery_bridge import _artifact_rows
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_targeted_recovery_bridge import (
    _batch_with_prefixes,
    _generate_enforced,
    _load_parent,
    _load_router,
)
from .capability_compiler_phase3_weak_residual import (
    EXPECTED_PARAMETERS,
    SharedWeakResidual,
    WEAK_CAPABILITIES,
    _attach,
    _parameter_count,
    _set_routes,
    _state_hash,
)
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-final-controls/1"
SYSTEMS = ("A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document = _json(path)
    if document.get("format") == "abi-capability-compiler-phase3-final-controls-repair/1":
        if document.get("status") != "PREREGISTERED_ZERO_IDENTITY_DERANGEMENT_REPAIR":
            raise Phase3Error("final-control repair governance changed")
        base_path = root / str(document["base_protocol"])
        if not base_path.is_file() or sha256_file(base_path) != document["base_protocol_sha256"]:
            raise Phase3Error("final-control base protocol changed")
        protocol = copy.deepcopy(_json(base_path))
        protocol["status"] = document["effective_status"]
        protocol["version"] = document["version"]
        protocol["control_outputs"] = copy.deepcopy(document["control_outputs"])
        protocol["bindings"].update(document["binding_overrides"])
    else:
        protocol = document
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") not in {
            "PREREGISTERED_FINAL_LINEAGE_MATCHED_CAUSAL_MATRIX",
            "PREREGISTERED_FINAL_LINEAGE_MATCHED_CAUSAL_MATRIX_ZERO_IDENTITY_REPAIR",
        }
        or tuple(protocol.get("systems", ())) != SYSTEMS
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("final-control governance changed")
    for relative, expected in protocol["bindings"].items():
        target = Path(relative) if Path(relative).is_absolute() else root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"final-control binding changed: {relative}")
    base_path = root / protocol["base_A0_protocol"]
    base = _json(base_path)
    if sha256_file(base_path) != protocol["bindings"][protocol["base_A0_protocol"]]:
        raise Phase3Error("A0 protocol changed")
    return protocol, sha256_file(path), base


def _prompt_hash_route(row: Mapping[str, Any]) -> int:
    count = int(row["prompt_tokens"])
    payload = ",".join(str(int(value)) for value in row["input_ids"][:count]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(WEAK_CAPABILITIES)


def _control_routes(system: str, rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    mapping = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    if system == "A1_label_free":
        values = [_prompt_hash_route(row) for row in rows]
    elif system == "A4_monolithic":
        values = [0 for _ in rows]
    else:
        values = [mapping[str(row["capability"])] for row in rows]
    return torch.tensor(values, dtype=torch.long, device=device)


def _derange_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["capability"]), str(row["view"]))
        grouped.setdefault(key, []).append(row)
    target_by_record: dict[str, tuple[Mapping[str, Any], int]] = {}
    for key in sorted(grouped):
        values = grouped[key]
        if len(values) < 2:
            raise Phase3Error("shuffled-target stratum depth changed")
        response = lambda row: tuple(row["input_ids"][int(row["prompt_tokens"]):])
        ordered = sorted(values, key=lambda row: (response(row), str(row["record_id"])))
        frequencies = Counter(response(row) for row in ordered)
        offset = max(frequencies.values())
        if offset * 2 > len(ordered):
            raise Phase3Error("no zero-identity target derangement exists")
        for index, row in enumerate(ordered):
            target = ordered[(index + offset) % len(ordered)]
            if response(row) == response(target):
                raise Phase3Error("constructed target derangement contains an identity")
            target_by_record[str(row["record_id"])] = (target, offset)
    result = []
    for row in rows:
        target, offset = target_by_record[str(row["record_id"])]
        prompt_count = int(row["prompt_tokens"])
        target_prompt = int(target["prompt_tokens"])
        prompt_ids = list(row["input_ids"][:prompt_count])
        response_ids = list(target["input_ids"][target_prompt:])
        if list(row["input_ids"][prompt_count:]) == response_ids:
            raise Phase3Error("target derangement contains an identity")
        changed = copy.deepcopy(dict(row))
        changed["input_ids"] = prompt_ids + response_ids
        changed["labels"] = [-100] * len(prompt_ids) + response_ids
        changed["response_tokens"] = len(response_ids)
        changed["deranged_target_record_id"] = str(target["record_id"])
        changed["derangement_offset"] = offset
        result.append(changed)
    if len(result) != len(rows):
        raise Phase3Error("target derangement lost rows")
    return result


@torch.inference_mode()
def _prefixes(
    system: str,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    horizon: int,
    device: torch.device,
) -> list[list[int]]:
    prefixes = []
    model.eval()
    for row in rows:
        prompt_count = int(row["prompt_tokens"])
        prompt = list(row["input_ids"][:prompt_count])
        _set_routes(model, _control_routes(system, [row], device))
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        task = torch.tensor([int(row["route"])], dtype=torch.long, device=device)
        result = model(
            ids,
            prompt_lengths=torch.tensor([prompt_count], dtype=torch.long, device=device),
            task_routes=task,
            use_cache=True,
        )
        cache, logits, generated = result["past_key_values"], result["logits"][:, -1], []
        for _ in range(horizon):
            selected = logits.argmax(dim=-1)
            token = int(selected.item())
            if token == int(tokenizer.eos_token_id):
                break
            generated.append(token)
            result = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True)
            cache, logits = result["past_key_values"], result["logits"][:, -1]
        prefixes.append(generated)
    model.train()
    return prefixes


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    model, tokenizer, _ = _load_parent(root, base, torch.device("cpu"))
    rows = _artifact_rows(root / base["supervision"]["artifact"])
    catalog = _json(root / base["supervision"]["source_catalog"])["probes"]
    examples = dual_examples(rows, catalog, tokenizer, int(base["training"]["max_tokens"]))
    deranged = _derange_targets(examples)
    residual = SharedWeakResidual()
    residual.load_state_dict(load_file(str(root / base["initialization"]["checkpoint"]), device="cpu"), strict=True)
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "systems": list(SYSTEMS),
        "common_parent_parameters": sum(value.numel() for value in model.parameters()),
        "matched_bridge_parameters": _parameter_count(residual),
        "dual_view_examples": len(examples),
        "deranged_examples": len(deranged),
        "target_derangement_identities": 0,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, system: str, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    if system not in SYSTEMS or output.exists():
        raise Phase3Error("invalid system or immutable control output exists")
    if not torch.cuda.is_available():
        raise Phase3Error("matched-control CUDA unavailable")
    cfg = base["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, base, device)
    residual = SharedWeakResidual().to(device)
    initialization = root / base["initialization"]["checkpoint"]
    residual.load_state_dict(load_file(str(initialization), device="cuda"), strict=True)
    initial_hash = _state_hash(residual.state_dict())
    handles = _attach(model, residual)
    rows = _artifact_rows(root / base["supervision"]["artifact"])
    catalog = _json(root / base["supervision"]["source_catalog"])["probes"]
    examples = dual_examples(rows, catalog, tokenizer, int(cfg["max_tokens"]))
    if system == "A2_shuffled":
        examples = _derange_targets(examples)
    sampler = DualViewSampler(examples, seed)
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )
    parent_before = _state_hash(model.state_dict())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    counts, recovery_counts, horizons = Counter(), Counter(), Counter()
    sequence = hashlib.sha256()
    teacher_tokens = prefix_tokens = recovery_batches = 0
    curves = []
    started = time.perf_counter()
    residual.train()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.teacher_forced_batch()
        prefixes: list[list[int]] = [[] for _ in selected]
        horizon = None
        if step >= int(cfg["recovery_start_step"]) and (step - int(cfg["recovery_start_step"])) % int(cfg["recovery_interval"]) == 0:
            recovery = sampler.recovery_batch(int(cfg["recovery_batch_size"]))
            choices = tuple(int(value) for value in cfg["recovery_horizons"])
            horizon = choices[recovery_batches % len(choices)]
            generated = _prefixes(system, model, tokenizer, recovery, horizon, device)
            by_stratum = {
                (str(row["capability"]), int(row["builder"]), str(row["view"])): (row, prefix)
                for row, prefix in zip(recovery, generated)
            }
            selected_keys = {
                (str(row["capability"]), int(row["builder"]), str(row["view"])): index
                for index, row in enumerate(selected)
            }
            for key, replacement in by_stratum.items():
                index = selected_keys[key]
                selected[index], prefixes[index] = replacement
                recovery_counts[":".join(map(str, key))] += 1
            recovery_batches += 1
            horizons[str(horizon)] += 1
            prefix_tokens += sum(len(value) for value in generated)
        ids, labels, attention, prompt_lengths, task_routes = _batch_with_prefixes(
            selected, int(tokenizer.eos_token_id), device, prefixes
        )
        _set_routes(model, _control_routes(system, selected, device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=task_routes,
                use_cache=False,
            )
            if system == "A3_bridge_only":
                loss = result["logits"].sum() * 0.0
            else:
                loss = _equal_record_prompt_overlap_ce(
                    result["logits"], labels, ids, prompt_lengths,
                    overlap_weight=float(cfg["prompt_overlap_weight"]),
                )
        loss.backward()
        if system != "A3_bridge_only":
            torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
        for row in selected:
            key = f"{row['capability']}:{row['builder']}:{row['view']}"
            counts[key] += 1
            if system != "A3_bridge_only":
                teacher_tokens += int(row["response_tokens"])
            sequence.update((str(row["record_id"]) + "\n").encode())
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"system": system, "step": step, "loss": float(loss.detach()), "recovery_horizon": horizon, "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    parent_after = _state_hash(model.state_dict())
    trained_hash = _state_hash(residual.state_dict())
    for handle in handles:
        handle.remove()
    if parent_before != parent_after:
        raise Phase3Error("frozen V463 parent changed")
    if system == "A3_bridge_only" and initial_hash != trained_hash:
        raise Phase3Error("bridge-only control changed without response loss")
    output.mkdir(parents=True)
    checkpoint = output / "control_bridge.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in residual.state_dict().items()},
        str(checkpoint),
        metadata={"format": FORMAT, "system": system},
    )
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_MATCHED_CONTROL_DEVELOPMENT_ONLY",
        "system": system,
        "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "initialization": {"sha256": sha256_file(initialization), "state_sha256": initial_hash},
        "parent": {"checkpoint_sha256": base["parent"]["checkpoint_sha256"], "mutated": False},
        "bridge_parameters": EXPECTED_PARAMETERS,
        "training": {
            "seed": seed,
            "steps": int(cfg["steps"]),
            "observations": sum(counts.values()),
            "observations_by_stratum": dict(sorted(counts.items())),
            "record_sequence_sha256": sequence.hexdigest(),
            "teacher_response_tokens_in_loss": teacher_tokens,
            "recovery_batches": recovery_batches,
            "recovery_by_stratum": dict(sorted(recovery_counts.items())),
            "recovery_horizon_batches": dict(sorted(horizons.items())),
            "autonomous_prefix_tokens": prefix_tokens,
            "wall_seconds": wall,
            "peak_process_rss_bytes": int(peak_rss),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
        },
        "teacher_present": False,
        "source_parameters_copied": 0,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


@torch.inference_mode()
def _generate_weak(
    system: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    maximum: int,
    capability: str,
    device: torch.device,
) -> tuple[str, list[int], int]:
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    row = {"prompt_tokens": len(prompt_ids), "input_ids": prompt_ids, "capability": capability}
    _set_routes(model, _control_routes(system, [row], device))
    task = torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device), task_routes=task, use_cache=True)
    cache, logits, generated = result["past_key_values"], result["logits"][:, -1], []
    for _ in range(maximum):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True)
        cache, logits = result["past_key_values"], result["logits"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated, int(task.item())


def evaluate(root: Path, protocol_path: Path, system: str, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    if system not in SYSTEMS or output.exists():
        raise Phase3Error("invalid system or immutable evaluation output exists")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / metadata["checkpoint"]["path"]
    if metadata["system"] != system or metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("control candidate lineage changed")
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, base, device)
    residual = SharedWeakResidual().to(device)
    residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True)
    residual.eval()
    handles = _attach(model, residual)
    router, router_tokenizer, router_protocol = _load_router(root, base)
    markers = artifact_markers(root / protocol["guard"]["artifact"])
    clause = str(protocol["guard"]["canonical_abstention_clause"])
    probes = development_probes(root / base["development"]["catalog_path"])
    parent = {str(row["probe_id"]): row for row in map(json.loads, (root / base["parent"]["development_outputs"]).open(encoding="utf-8"))}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        capability = str(probe["canonical_capability"])
        routed, details = sparse._route(router, router_tokenizer, router_protocol, prompt)
        if capability in WEAK_CAPABILITIES:
            original, _, task_route = _generate_weak(system, model, tokenizer, prompt, int(probe["max_new_tokens"]), capability, device)
            value, terminated = truncate_at_first_v2_collapse(original)
            prefixed = False
            if capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers):
                value = clause + (" " + value if value else "")
                prefixed = True
            tokens = [int(value) for value in tokenizer.encode(value, add_special_tokens=False)]
        else:
            value, tokens, task_route = _generate_enforced(model, tokenizer, prompt, int(probe["max_new_tokens"]), capability, device)
            original, terminated, prefixed = value, False, False
        rows.append({
            "probe_id": str(probe["probe_id"]),
            "capability": capability,
            "output": value,
            "original_output": original,
            "output_token_ids": tokens,
            "automatic_capability_route": routed,
            "capability_route_correct": routed == capability,
            "control_residual_route": None if capability not in WEAK_CAPABILITIES else int(_control_routes(system, [{"prompt_tokens": len(tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)), "input_ids": tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False), "capability": capability}], torch.device("cpu"))[0]),
            "task_route": task_route,
            "strong_parent_output_exact": None if capability in WEAK_CAPABILITIES else value == str(parent[str(probe["probe_id"])]["output"]),
            "guard_terminated": terminated,
            "abstention_clause_prefixed": prefixed,
            "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
            "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability),
            "repetition_collapse_v2": repetition_collapse_v2(value),
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"system": system, "evaluated": index + 1}), flush=True)
    for handle in handles:
        handle.remove()
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passed = sum(row["functional_pass_v1"] for row in values)
        per[capability] = {"passes_v1": passed, "observations": len(values), "wilson_v1": wilson(passed, len(values)), "collapses_v2": sum(row["repetition_collapse_v2"] for row in values)}
    result = {
        "format": "abi-capability-compiler-phase3-final-control-evaluation/1",
        "status": "COMPLETE_MATCHED_CONTROL_DEVELOPMENT_EVALUATION",
        "system": system,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows),
        "functional_passes_v2": sum(row["functional_pass_v2"] for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in rows),
        "router_correct": sum(row["capability_route_correct"] for row in rows),
        "strong_routes_exact": sum(row["strong_parent_output_exact"] is True for row in rows),
        "guard_terminations": sum(row["guard_terminated"] for row in rows),
        "abstention_prefixes": sum(row["abstention_clause_prefixed"] for row in rows),
        "raw_outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def decide(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable final-control decision exists")
    a0 = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["A0_outputs"]).open(encoding="utf-8"))}
    if len(a0) != 1400:
        raise Phase3Error("A0 output depth changed")
    comparisons, systems = {}, {}
    for index, system in enumerate(SYSTEMS):
        directory = root / protocol["control_outputs"][system]
        result = _json(directory / "result.json")
        rows = {str(row["probe_id"]): row for row in map(json.loads, (directory / "development_outputs.jsonl").open(encoding="utf-8"))}
        if set(rows) != set(a0) or result["system"] != system:
            raise Phase3Error("A0/control pairing changed")
        paired = []
        for probe_id, candidate in a0.items():
            control = rows[probe_id]
            if candidate["capability"] != control["capability"]:
                raise Phase3Error("A0/control capability join changed")
            paired.append({"capability": candidate["capability"], "candidate_pass": bool(candidate["functional_pass_v1"]), "teacher_pass": bool(control["functional_pass_v1"])})
        comparison = paired_stratified_bootstrap(paired, replicates=10_000, seed=5181729 + index)
        comparisons[system] = comparison
        systems[system] = {"functional_passes_v1": result["functional_passes_v1"], "repetition_collapses_v2": result["repetition_collapses_v2"], "checkpoint_sha256": result["checkpoint_sha256"], "outputs_sha256": result["raw_outputs_sha256"]}
    gates = {system: comparison["lower_95"] > 0.0 for system, comparison in comparisons.items()}
    result = {
        "format": "abi-capability-compiler-phase3-final-controls-decision/1",
        "status": "PASS_FINAL_LINEAGE_MATCHED_CAUSAL_CONTROLS" if all(gates.values()) else "FAIL_FINAL_LINEAGE_CAUSAL_CONTROL_GATE",
        "protocol_sha256": protocol_sha,
        "A0": {"functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in a0.values()), "checkpoint_sha256": protocol["A0_checkpoint_sha256"], "outputs_sha256": sha256_file(root / protocol["A0_outputs"])},
        "systems": systems,
        "paired_A0_minus_control": comparisons,
        "gates": gates,
        "all_controls_passed": all(gates.values()),
        "final_test_accessed": False,
        "historical_evidence_changed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CONTROLS_PROTOCOL_V518.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--system", choices=SYSTEMS, required=True)
    train_parser.add_argument("--output-dir", required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--system", choices=SYSTEMS, required=True)
    eval_parser.add_argument("--candidate-dir", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    decide_parser = sub.add_parser("decide")
    decide_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root, protocol = Path.cwd().resolve(), Path.cwd().resolve() / args.protocol
    if args.command == "preflight":
        result = preflight(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, args.system, root / args.output_dir)
    elif args.command == "evaluate":
        result = evaluate(root, protocol, args.system, root / args.candidate_dir, root / args.output_dir)
    else:
        result = decide(root, protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
