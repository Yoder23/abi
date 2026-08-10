"""Targeted weak-capability support plus autonomous-prefix recovery bridge."""

from __future__ import annotations

from collections import Counter
import argparse
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
from .capability_compiler_phase3 import (
    CAPABILITY_TO_ROUTE,
    Phase3Error,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_sequence_bridge import _examples
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443
from .capability_compiler_phase3_weak_residual import (
    EXPECTED_PARAMETERS,
    SharedWeakResidual,
    WEAK_CAPABILITIES,
    WeakBalancedSampler,
    _attach,
    _parameter_count,
    _set_routes,
    _state_hash,
)
from .capability_compiler_phase3_weak_support_audit import (
    _load_verified_acquisition_ir,
)
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-targeted-recovery-bridge/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SINGLE_MIXED_SUPPORT_CAUSAL_RECOVERY_BRIDGE"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or tuple(protocol.get("architecture", {}).get("weak_capabilities", ()))
        != WEAK_CAPABILITIES
        or int(protocol.get("architecture", {}).get("trainable_parameters", -1))
        != EXPECTED_PARAMETERS
    ):
        raise Phase3Error("targeted-recovery governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"targeted-recovery binding changed: {relative}")
    return protocol, sha256_file(path)


def _batch_with_prefixes(
    rows: Sequence[Mapping[str, Any]],
    eos: int,
    device: torch.device,
    generated_prefixes: Sequence[Sequence[int]] | None = None,
):
    encoded = []
    targets = []
    prompt_lengths = []
    routes = []
    for index, row in enumerate(rows):
        prompt_count = int(row["prompt_tokens"])
        prompt = list(row["input_ids"][:prompt_count])
        response = list(row["input_ids"][prompt_count:])
        generated = [] if generated_prefixes is None else list(generated_prefixes[index])
        generated_count = min(len(generated), max(0, len(response) - 1))
        sequence = prompt + generated[:generated_count] + response[generated_count:]
        target = [-100] * len(prompt) + response
        encoded.append(sequence)
        targets.append(target[: len(sequence)])
        prompt_lengths.append(prompt_count)
        routes.append(int(row["route"]))
    width = max(len(row) for row in encoded)
    ids = torch.full((len(rows), width), eos, dtype=torch.long, device=device)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, (sequence, target) in enumerate(zip(encoded, targets)):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        labels[index, : len(target)] = torch.tensor(target, dtype=torch.long, device=device)
        attention[index, : len(sequence)] = 1
    return (
        ids,
        labels,
        attention,
        torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        torch.tensor(routes, dtype=torch.long, device=device),
    )


def _weak_routes(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    mapping = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    return torch.tensor(
        [mapping[str(row["capability"])] for row in rows],
        dtype=torch.long,
        device=device,
    )


@torch.inference_mode()
def _autonomous_prefixes(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    horizon: int,
    device: torch.device,
) -> list[list[int]]:
    prefixes = []
    model.eval()
    weak_to_id = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    for row in rows:
        capability = str(row["capability"])
        prompt_count = int(row["prompt_tokens"])
        prompt = list(row["input_ids"][:prompt_count])
        weak_route = weak_to_id[capability]
        task_route = int(row["route"])
        _set_routes(model, torch.tensor([weak_route], dtype=torch.long, device=device))
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        route_tensor = torch.tensor([task_route], dtype=torch.long, device=device)
        result = model(
            ids,
            prompt_lengths=torch.tensor([prompt_count], dtype=torch.long, device=device),
            task_routes=route_tensor,
            use_cache=True,
        )
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
        generated = []
        for _ in range(horizon):
            selected = logits.argmax(dim=-1)
            token = int(selected.item())
            if token == int(tokenizer.eos_token_id):
                break
            generated.append(token)
            result = model(
                selected[:, None],
                task_routes=route_tensor,
                past_key_values=cache,
                use_cache=True,
            )
            cache = result["past_key_values"]
            logits = result["logits"][:, -1]
        prefixes.append(generated)
    model.train()
    return prefixes


def _load_parent(root: Path, protocol: dict[str, Any], device: torch.device):
    return _load_v443(root, protocol, device)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, _ = _load_parent(root, protocol, torch.device("cpu"))
    residual = SharedWeakResidual()
    targeted = _load_verified_acquisition_ir(root / protocol["targeted_ir"]["path"])
    original = load_phase1_ir(root / protocol["anchor_ir"]["path"])
    target_counts = Counter(row["capability"] for row in targeted if row["capability"] in WEAK_CAPABILITIES)
    anchor_counts = Counter(row["capability"] for row in original if row["capability"] in WEAK_CAPABILITIES)
    if target_counts != Counter({name: 500 for name in WEAK_CAPABILITIES}) or anchor_counts != Counter({name: 500 for name in WEAK_CAPABILITIES}):
        raise Phase3Error("target or anchor weak-capability depth changed")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "frozen_parent_parameters": sum(value.numel() for value in model.parameters()),
        "bridge_parameters": _parameter_count(residual),
        "targeted_records": sum(target_counts.values()),
        "anchor_records": sum(anchor_counts.values()),
        "tokenizer_vocabulary": len(tokenizer),
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable targeted-recovery output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("targeted-recovery CUDA unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    handles = _attach(model, residual)
    targeted_rows = _load_verified_acquisition_ir(root / protocol["targeted_ir"]["path"])
    anchor_rows = load_phase1_ir(root / protocol["anchor_ir"]["path"])
    targeted_examples = [
        row
        for row in _examples(targeted_rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
        if row["capability"] in WEAK_CAPABILITIES
    ]
    anchor_examples = [
        row
        for row in _examples(anchor_rows, tokenizer, system="A0", seed=seed + 1, max_tokens=int(cfg["max_tokens"]))
        if row["capability"] in WEAK_CAPABILITIES
    ]
    targeted_sampler = WeakBalancedSampler(targeted_examples, seed)
    anchor_sampler = WeakBalancedSampler(anchor_examples, seed + 1)
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
    targeted_counts = Counter()
    anchor_counts = Counter()
    sequence_sha = hashlib.sha256()
    targeted_tokens = 0
    anchor_tokens = 0
    recovery_prefix_tokens = 0
    recovery_batches = 0
    recovery_horizon_counts = Counter()
    curves = []
    started = time.perf_counter()
    for step in range(1, int(cfg["steps"]) + 1):
        targeted = targeted_sampler.batch(int(cfg["targeted_batch_size"]))
        anchors = anchor_sampler.batch(int(cfg["anchor_batch_size"]))
        prefixes: list[list[int]] = [[] for _ in targeted]
        recovery_horizon = None
        if step >= int(cfg["recovery_start_step"]) and (step - int(cfg["recovery_start_step"])) % int(cfg["recovery_interval"]) == 0:
            horizons = tuple(int(value) for value in cfg["recovery_horizons"])
            recovery_horizon = horizons[recovery_batches % len(horizons)]
            prefixes = _autonomous_prefixes(model, tokenizer, targeted, recovery_horizon, device)
            recovery_batches += 1
            recovery_horizon_counts[str(recovery_horizon)] += 1
            recovery_prefix_tokens += sum(len(value) for value in prefixes)
        selected = [*targeted, *anchors]
        generated_prefixes = [*prefixes, *([[]] * len(anchors))]
        ids, labels, attention, prompt_lengths, task_routes = _batch_with_prefixes(
            selected,
            int(tokenizer.eos_token_id),
            device,
            generated_prefixes,
        )
        _set_routes(model, _weak_routes(selected, device))
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
        for row in targeted:
            targeted_counts[str(row["capability"])] += 1
            targeted_tokens += int(row["response_tokens"])
            sequence_sha.update(b"T:" + str(row["record_id"]).encode() + b"\n")
        for row in anchors:
            anchor_counts[str(row["capability"])] += 1
            anchor_tokens += int(row["response_tokens"])
            sequence_sha.update(b"A:" + str(row["record_id"]).encode() + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {
                "step": step,
                "language_loss": float(loss.detach()),
                "recovery_horizon": recovery_horizon,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    residual.eval()
    parent_after = _state_hash(model.state_dict())
    for handle in handles:
        handle.remove()
    if parent_before != parent_after:
        raise Phase3Error("frozen V463 parent changed")
    output.mkdir(parents=True)
    checkpoint = output / "targeted_recovery_bridge.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in residual.state_dict().items()},
        str(checkpoint),
        metadata={"format": FORMAT},
    )
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL",
        "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "state_sha256_before": parent_before, "state_sha256_after": parent_after, "mutated": False},
        "bridge": {"parameters": EXPECTED_PARAMETERS, "weak_capabilities": WEAK_CAPABILITIES, "shared_across_blocks": True, "source_parameters_copied": 0},
        "training": {
            "device": "cuda",
            "seed": seed,
            "steps": int(cfg["steps"]),
            "targeted_observations": sum(targeted_counts.values()),
            "anchor_observations": sum(anchor_counts.values()),
            "targeted_by_capability": dict(sorted(targeted_counts.items())),
            "anchor_by_capability": dict(sorted(anchor_counts.items())),
            "targeted_teacher_tokens_seen": targeted_tokens,
            "anchor_teacher_tokens_seen": anchor_tokens,
            "autonomous_prefix_tokens_seen": recovery_prefix_tokens,
            "recovery_batches": recovery_batches,
            "recovery_horizon_batches": dict(sorted(recovery_horizon_counts.items())),
            "record_sequence_sha256": sequence_sha.hexdigest(),
            "wall_seconds": wall,
            "active_parameter_seconds": EXPECTED_PARAMETERS * wall,
            "peak_process_rss_bytes": int(peak_rss),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
        },
        "teacher_present_during_training": False,
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def _load_router(root: Path, protocol: dict[str, Any]):
    router_protocol = _json(root / protocol["router"]["protocol"])
    router, tokenizer = sparse._load(root, router_protocol, root / protocol["router"]["candidate_dir"])
    return router, tokenizer, router_protocol


@torch.inference_mode()
def _generate_enforced(
    model: Any,
    tokenizer: Any,
    prompt: str,
    maximum: int,
    capability: str,
    device: torch.device,
):
    weak_to_id = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    weak_route = weak_to_id.get(capability, -1)
    _set_routes(model, torch.tensor([weak_route], dtype=torch.long, device=device))
    forced = (
        torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long, device=device)
        if weak_route >= 0
        else None
    )
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device),
        task_routes=forced,
        use_cache=True,
    )
    route = result["task_routes"].detach().clone()
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    generated = []
    for _ in range(maximum):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=route, past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated, int(route.item())


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable targeted-recovery evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / metadata["checkpoint"]["path"]
    if metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("targeted-recovery lineage changed")
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True)
    residual.eval()
    handles = _attach(model, residual)
    router, router_tokenizer, router_protocol = _load_router(root, protocol)
    probes = development_probes(root / protocol["development"]["catalog_path"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}
    parent_rows = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["parent"]["development_outputs"]).open(encoding="utf-8"))}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        routed, details = sparse._route(router, router_tokenizer, router_protocol, prompt)
        value, tokens, task_route = _generate_enforced(model, tokenizer, prompt, int(probe["max_new_tokens"]), routed, device)
        capability = str(probe["canonical_capability"])
        rows.append({
            "probe_id": str(probe["probe_id"]),
            "capability": capability,
            "output": value,
            "output_token_ids": tokens,
            "automatic_capability_route": routed,
            "capability_route_correct": routed == capability,
            "task_route": task_route,
            "weak_route_active": routed in WEAK_CAPABILITIES,
            "router_segment_count": len(details),
            "strong_parent_output_exact": None if capability in WEAK_CAPABILITIES else value == str(parent_rows[str(probe["probe_id"])]["output"]),
            "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
            "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability),
            "repetition_collapse_v2": repetition_collapse_v2(value),
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    for handle in handles:
        handle.remove()
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        v1 = sum(row["functional_pass_v1"] for row in values)
        v2 = sum(row["functional_pass_v2"] for row in values)
        per[capability] = {"passes_v1": v1, "passes_v2": v2, "observations": len(values), "v2_collapses": sum(row["repetition_collapse_v2"] for row in values), "wilson_v1": wilson(v1, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    a = protocol["absolute_screen"]
    collapses = sum(row["repetition_collapse_v2"] for row in rows)
    strong = [row for row in rows if row["capability"] not in WEAK_CAPABILITIES]
    gates = {
        "qualified_router_exact": all(row["capability_route_correct"] for row in rows),
        "strong_routes_byte_exact_to_v463": all(row["strong_parent_output_exact"] is True for row in strong),
        "per_capability_functional_v1": all(value["wilson_v1"]["point"] >= float(a["per_capability_functional_point_estimate_minimum"]) and value["wilson_v1"]["lower_95"] >= float(a["per_capability_functional_wilson_lower_minimum"]) for value in per.values()),
        "critical_capabilities_v1": all(per[name]["wilson_v1"]["point"] >= float(a["critical_point_minimum"]) and per[name]["wilson_v1"]["lower_95"] >= float(a["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_v2_repetition_collapses": collapses <= int(a["repetition_collapse_v2_count_maximum"]),
        "teacher_relative_noninferiority_v1": relative["lower_95"] >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "frozen_parent": metadata["parent"]["mutated"] is False,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-targeted-recovery-bridge-result/1",
        "status": "PASS_INITIAL_TARGETED_RECOVERY_SCREEN_REPLICATION_RUNTIME_OPEN" if passed else "FAIL_TARGETED_RECOVERY_BRIDGE_CLOSED",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows),
        "functional_passes_v2": sum(row["functional_pass_v2"] for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v2": collapses,
        "strong_routes_exact": sum(row["strong_parent_output_exact"] is True for row in strong),
        "strong_route_observations": len(strong),
        "router_correct": sum(row["capability_route_correct"] for row in rows),
        "teacher_comparison_v1": relative,
        "gates": gates,
        "passed": passed,
        "raw_outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "claim_boundary": "Single development-only targeted support and causal-recovery bridge; replication, CPU runtime, TTFT, RSS, final quality, minimum information, and Phase 3 remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TARGETED_RECOVERY_BRIDGE_PROTOCOL_V473.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    tp = sub.add_parser("train"); tp.add_argument("--output-dir", required=True)
    ep = sub.add_parser("evaluate"); ep.add_argument("--candidate-dir", required=True); ep.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
