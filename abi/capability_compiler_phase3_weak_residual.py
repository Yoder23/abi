"""Bounded four-capability corrective residual on the sealed V463 host."""

from __future__ import annotations

import argparse
from collections import Counter
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
import torch.nn.functional as F
from torch import nn

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
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_sequence_bridge import _batch, _examples, _generate
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-weak-residual/1"
WEAK_CAPABILITIES = (
    "abstention",
    "coherence",
    "fluent_realization",
    "tone_control",
)
WIDTH = 768
RANK = 64
EXPECTED_PARAMETERS = 100_352


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SINGLE_BOUNDED_CORRECTIVE_BRIDGE"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or tuple(protocol.get("architecture", {}).get("weak_capabilities", ()))
        != WEAK_CAPABILITIES
        or int(protocol.get("architecture", {}).get("trainable_parameters", -1))
        != EXPECTED_PARAMETERS
    ):
        raise Phase3Error("weak-residual governance changed")
    for relative, expected in protocol["bindings"].items():
        target = Path(relative) if Path(relative).is_absolute() else root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"weak-residual binding changed: {relative}")
    return protocol, sha256_file(path)


class SharedWeakResidual(nn.Module):
    """One nonlinear residual reused at all blocks with four tiny route codes."""

    def __init__(self, width: int = WIDTH, rank: int = RANK, routes: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.route_scale = nn.Embedding(routes, rank)
        self.route_shift = nn.Embedding(routes, rank)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.route_scale.weight)
        nn.init.zeros_(self.route_shift.weight)

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        low = self.down(self.norm(hidden))
        scale = 1.0 + torch.tanh(self.route_scale(routes))
        shift = self.route_shift(routes)
        return self.up(F.silu(low * scale[:, None, :] + shift[:, None, :]))


def _parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


def _set_routes(model: nn.Module, routes: torch.Tensor) -> None:
    routes = routes.long().flatten()
    for block in model.transformer.h:
        block._abi_weak_capability_routes = routes


def _hook(residual: SharedWeakResidual):
    def apply(module, args, kwargs):
        hidden = args[0]
        routes = getattr(module, "_abi_weak_capability_routes", None)
        if routes is None or routes.shape[0] != hidden.shape[0]:
            raise Phase3Error("weak capability route is absent or malformed")
        active = routes.ge(0)
        if not bool(active.any()):
            return args, kwargs
        selected = routes.clamp_min(0)
        delta = residual.delta(hidden, selected)
        delta = delta * active.to(delta.dtype)[:, None, None]
        return (hidden + delta, *args[1:]), kwargs

    return apply


def _attach(model: nn.Module, residual: SharedWeakResidual):
    return [
        block.register_forward_pre_hook(_hook(residual), with_kwargs=True)
        for block in model.transformer.h
    ]


class WeakBalancedSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int) -> None:
        import random

        self.grouped = {
            name: [row for row in rows if row["capability"] == name]
            for name in WEAK_CAPABILITIES
        }
        if any(not values for values in self.grouped.values()):
            raise Phase3Error("weak capability training evidence is incomplete")
        self.rng = random.Random(seed)
        self.index = 0

    def batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            name = WEAK_CAPABILITIES[self.index % len(WEAK_CAPABILITIES)]
            self.index += 1
            values = self.grouped[name]
            result.append(values[self.rng.randrange(len(values))])
        return result


def _weak_route_tensor(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    mapping = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    return torch.tensor(
        [mapping[str(row["capability"])] for row in rows],
        dtype=torch.long,
        device=device,
    )


def _load_parent(root: Path, protocol: dict[str, Any], device: torch.device):
    model, tokenizer, metadata = _load_v443(root, protocol, device)
    if sum(value.numel() for value in model.parameters()) != int(
        protocol["parent"]["parameters"]
    ):
        raise Phase3Error("V463 parent parameter count changed")
    return model, tokenizer, metadata


def _load_router(root: Path, protocol: dict[str, Any]):
    router_protocol = _json(root / protocol["router"]["protocol"])
    candidate = root / protocol["router"]["candidate_dir"]
    return sparse._load(root, router_protocol, candidate), router_protocol


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_parent(root, protocol, torch.device("cpu"))
    residual = SharedWeakResidual()
    if _parameter_count(residual) != EXPECTED_PARAMETERS:
        raise Phase3Error("weak residual parameter count changed")
    strong = torch.full((2,), -1, dtype=torch.long)
    hidden = torch.randn(2, 5, WIDTH)
    before = hidden.clone()
    block = model.transformer.h[0]
    block._abi_weak_capability_routes = strong
    returned, _ = _hook(residual)(block, (hidden,), {})
    if not torch.equal(returned[0], before):
        raise Phase3Error("strong-route bypass is not exact")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "frozen_parent_parameters": sum(value.numel() for value in model.parameters()),
        "new_trainable_bridge_parameters": _parameter_count(residual),
        "shared_block_invocations": len(model.transformer.h),
        "strong_route_exact_bypass": True,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable weak-residual output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("weak-residual CUDA unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    if _parameter_count(residual) != EXPECTED_PARAMETERS:
        raise Phase3Error("weak residual parameter count changed")
    handles = _attach(model, residual)
    rows = load_phase1_ir(root / protocol["phase1_ir"]["path"])
    examples = _examples(
        rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"])
    )
    sampler = WeakBalancedSampler(examples, seed)
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
    sampled = Counter()
    sequence_sha = hashlib.sha256()
    teacher_tokens = 0
    curves = []
    started = time.perf_counter()
    residual.train()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.batch(int(cfg["batch_size"]))
        ids, labels, attention, prompt_lengths, task_routes = _batch(
            selected, int(tokenizer.eos_token_id), device
        )
        _set_routes(model, _weak_route_tensor(selected, device))
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
        torch.nn.utils.clip_grad_norm_(
            residual.parameters(), float(cfg["gradient_clip_norm"])
        )
        optimizer.step()
        for row in selected:
            sampled[str(row["capability"])] += 1
            sequence_sha.update(str(row["record_id"]).encode("ascii") + b"\n")
            teacher_tokens += int(row["response_tokens"])
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {
                "step": step,
                "language_loss": float(loss.detach()),
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
    checkpoint = output / "weak_residual.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in residual.state_dict().items()
        },
        str(checkpoint),
        metadata={"format": FORMAT},
    )
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL",
        "protocol_sha256": protocol_sha,
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "parent": {
            "checkpoint_sha256": protocol["parent"]["checkpoint_sha256"],
            "state_sha256_before": parent_before,
            "state_sha256_after": parent_after,
            "mutated": False,
        },
        "bridge": {
            "rank": RANK,
            "weak_capabilities": WEAK_CAPABILITIES,
            "parameters": EXPECTED_PARAMETERS,
            "shared_across_blocks": True,
            "source_parameters_copied": 0,
        },
        "training": {
            "device": "cuda",
            "seed": seed,
            "steps": int(cfg["steps"]),
            "batch_size": int(cfg["batch_size"]),
            "teacher_response_tokens_seen": teacher_tokens,
            "sampled_records_by_capability": dict(sorted(sampled.items())),
            "successful_record_sequence_sha256": sequence_sha.hexdigest(),
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
    metadata["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(metadata)
    ).hexdigest()
    _write_immutable(
        output / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n",
    )
    return metadata


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable weak-residual evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / metadata["checkpoint"]["path"]
    if (
        metadata["protocol_sha256"] != protocol_sha
        or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]
    ):
        raise Phase3Error("weak-residual lineage changed")
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True)
    residual.eval()
    handles = _attach(model, residual)
    (router, router_tokenizer), router_protocol = _load_router(root, protocol)
    probes = development_probes(root / protocol["development"]["catalog_path"])
    teacher = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"),
        )
    }
    parent_rows = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["parent"]["development_outputs"]).open(encoding="utf-8"),
        )
    }
    weak_to_id = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        routed, details = sparse._route(router, router_tokenizer, router_protocol, prompt)
        weak_route = weak_to_id.get(routed, -1)
        _set_routes(model, torch.tensor([weak_route], dtype=torch.long, device=device))
        value, tokens, task_route = _generate(
            model, tokenizer, prompt, int(probe["max_new_tokens"]), device
        )
        capability = str(probe["canonical_capability"])
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": capability,
                "output": value,
                "output_token_ids": tokens,
                "automatic_task_route": task_route,
                "automatic_capability_route": routed,
                "capability_route_correct": routed == capability,
                "router_segment_count": len(details),
                "weak_residual_active": weak_route >= 0,
                "strong_parent_output_exact": (
                    None
                    if capability in WEAK_CAPABILITIES
                    else value == str(parent_rows[str(probe["probe_id"])]["output"])
                ),
                "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(
                    value, probe["evaluator"], capability
                ),
                "repetition_collapse_v2": repetition_collapse_v2(value),
            }
        )
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
        passes_v1 = sum(row["functional_pass_v1"] for row in values)
        passes_v2 = sum(row["functional_pass_v2"] for row in values)
        per[capability] = {
            "passes_v1": passes_v1,
            "passes_v2": passes_v2,
            "observations": len(values),
            "v2_collapses": sum(row["repetition_collapse_v2"] for row in values),
            "wilson_v1": wilson(passes_v1, len(values)),
        }
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    paired = [
        {
            "capability": row["capability"],
            "candidate_pass": bool(row["functional_pass_v1"]),
            "teacher_pass": evaluate_functional(
                str(teacher[row["probe_id"]]["output"]),
                probe_by_id[row["probe_id"]]["evaluator"],
            ),
        }
        for row in rows
    ]
    relative = paired_stratified_bootstrap(
        paired,
        replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]),
        seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]),
    )
    a = protocol["absolute_screen"]
    collapses = sum(row["repetition_collapse_v2"] for row in rows)
    strong_rows = [row for row in rows if row["capability"] not in WEAK_CAPABILITIES]
    gates = {
        "qualified_router_exact": all(row["capability_route_correct"] for row in rows),
        "strong_routes_byte_exact_to_v463": all(
            row["strong_parent_output_exact"] is True for row in strong_rows
        ),
        "per_capability_functional_v1": all(
            value["wilson_v1"]["point"]
            >= float(a["per_capability_functional_point_estimate_minimum"])
            and value["wilson_v1"]["lower_95"]
            >= float(a["per_capability_functional_wilson_lower_minimum"])
            for value in per.values()
        ),
        "critical_capabilities_v1": all(
            per[name]["wilson_v1"]["point"] >= float(a["critical_point_minimum"])
            and per[name]["wilson_v1"]["lower_95"]
            >= float(a["critical_wilson_lower_minimum"])
            for name in ("prompt_grounding", "instruction_following", "abstention")
        ),
        "zero_v2_repetition_collapses": collapses
        <= int(a["repetition_collapse_v2_count_maximum"]),
        "teacher_relative_noninferiority_v1": relative["lower_95"]
        >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "frozen_parent": metadata["parent"]["mutated"] is False,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-weak-residual-result/1",
        "status": (
            "PASS_INITIAL_WEAK_RESIDUAL_SCREEN_REPLICATION_RUNTIME_OPEN"
            if passed
            else "FAIL_WEAK_RESIDUAL_CLOSED"
        ),
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows),
        "functional_passes_v2": sum(row["functional_pass_v2"] for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v2": collapses,
        "strong_routes_exact": sum(
            row["strong_parent_output_exact"] is True for row in strong_rows
        ),
        "strong_route_observations": len(strong_rows),
        "router_correct": sum(row["capability_route_correct"] for row in rows),
        "teacher_comparison_v1": relative,
        "gates": gates,
        "passed": passed,
        "raw_outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "claim_boundary": "Single development-only corrective bridge; replication, CPU runtime, TTFT, RSS, final quality, minimum information, and Phase 3 remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_WEAK_RESIDUAL_PROTOCOL_V465.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = root / args.protocol
    if args.command == "preflight":
        result = preflight(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, root / args.output_dir)
    else:
        result = evaluate(
            root, protocol, root / args.candidate_dir, root / args.output_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
