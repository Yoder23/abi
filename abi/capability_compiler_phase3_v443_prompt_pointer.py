"""Bounded selective prompt-identity carriage on the frozen V443 candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch

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
from .capability_compiler_phase3_qualified_transition_control import _load_parent, _state_hash
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_sequence_bridge import _BalancedSampler, _batch, _examples
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_full_core_acquisition import _balanced_prompt_identity_supervision_loss
from .layercake_host import PromptIdentityBridge, _prompt_identity_next_probabilities


FORMAT = "abi-capability-compiler-phase3-v443-prompt-pointer/1"
BRIDGE_RANK = 32
BRIDGE_ROUTES = 10
BRIDGE_PARAMETERS = 49_931


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SINGLE_BOUNDED_SUCCESSOR"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("V443 pointer governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V443 pointer binding changed: {name}")
    return protocol, sha256_file(path)


def _load_v443(root: Path, protocol: dict[str, Any], device: torch.device):
    base_protocol = _json(root / protocol["parent"]["base_protocol"])
    _, model, tokenizer, metadata = _load_parent(root, base_protocol, device)
    checkpoint = root / protocol["parent"]["checkpoint"]
    if sha256_file(checkpoint) != protocol["parent"]["checkpoint_sha256"]:
        raise Phase3Error("V443 parent checkpoint changed")
    model.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, metadata


def _bridge(device: torch.device) -> PromptIdentityBridge:
    bridge = PromptIdentityBridge(width=768, rank=BRIDGE_RANK, routes=BRIDGE_ROUTES).to(device)
    if sum(value.numel() for value in bridge.parameters()) != BRIDGE_PARAMETERS:
        raise Phase3Error("prompt pointer parameter count changed")
    return bridge


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_v443(root, protocol, torch.device("cpu"))
    bridge = _bridge(torch.device("cpu"))
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "frozen_parent_parameters": sum(value.numel() for value in model.parameters()),
        "new_trainable_bridge_parameters": sum(value.numel() for value in bridge.parameters()),
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable pointer output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("V443 pointer CUDA device unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_v443(root, protocol, device)
    bridge = _bridge(device)
    rows = load_phase1_ir(root / protocol["phase1_ir"]["path"])
    examples = _examples(rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    sampler = _BalancedSampler(examples, seed)
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    parent_before = _state_hash(model.state_dict())
    successful = 0
    language_tokens = 0
    sampled = Counter()
    sequence_sha = hashlib.sha256()
    curves = []
    started = time.perf_counter()
    bridge.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        ids, labels, attention, prompt_lengths, routes = _batch(
            selected, int(tokenizer.eos_token_id), device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=routes,
                use_cache=False,
            )
        with torch.autocast("cuda", dtype=torch.float16):
            loss = _balanced_prompt_identity_supervision_loss(
                hidden=result["hidden"].detach(),
                input_ids=ids,
                labels=labels,
                prompt_lengths=prompt_lengths,
                routes=routes,
                bridge=bridge,
                parent_logits=result["logits"].detach(),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(cfg["gradient_clip_norm"]))
        optimizer.step()
        successful += 1
        for row in selected:
            sequence_sha.update(str(row["record_id"]).encode("ascii") + b"\n")
            language_tokens += int(row["response_tokens"])
            sampled[str(row["capability"])] += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {
                "step": successful,
                "prompt_identity_loss": float(loss.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    bridge.eval()
    parent_after = _state_hash(model.state_dict())
    if parent_before != parent_after:
        raise Phase3Error("frozen V443 parent changed")
    output.mkdir(parents=True)
    checkpoint = output / "prompt_identity.safetensors"
    state = {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()}
    save_file(state, str(checkpoint), metadata={"format": FORMAT})
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL",
        "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "state_sha256_before": parent_before, "state_sha256_after": parent_after, "mutated": False},
        "bridge": {"rank": BRIDGE_RANK, "routes": BRIDGE_ROUTES, "parameters": BRIDGE_PARAMETERS, "source_parameters_copied": 0},
        "training": {
            "device": "cuda",
            "seed": seed,
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "learning_rate": float(cfg["learning_rate"]),
            "teacher_response_tokens_seen": language_tokens,
            "successful_record_sequence_sha256": sequence_sha.hexdigest(),
            "sampled_records_by_capability": dict(sorted(sampled.items())),
            "wall_seconds": wall,
            "active_parameter_seconds": BRIDGE_PARAMETERS * wall,
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


@torch.inference_mode()
def _generate(model, bridge, tokenizer, prompt: str, maximum: int, device: torch.device):
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), use_cache=True)
    route = result["task_routes"]
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    prompt_tensor = ids[0]
    prompt_keys = bridge.key(result["hidden"][0, : len(prompt_ids)])
    next_hidden = result["hidden"][:, -1]
    generated: list[int] = []
    for _ in range(maximum):
        probabilities = _prompt_identity_next_probabilities(
            logits=logits,
            query_hidden=next_hidden,
            prompt_keys=prompt_keys,
            prompt_ids=prompt_tensor,
            route=route,
            bridge=bridge,
        )
        selected = probabilities.argmax().reshape(1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=route, past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
        next_hidden = result["hidden"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated, int(route.item())


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable pointer evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / metadata["checkpoint"]["path"]
    if metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("pointer candidate lineage changed")
    device = torch.device("cuda")
    model, tokenizer, _ = _load_v443(root, protocol, device)
    bridge = _bridge(device)
    bridge.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True)
    bridge.eval()
    probes = development_probes(root / protocol["development"]["catalog_path"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        value, tokens, route = _generate(model, bridge, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device)
        rows.append({
            "probe_id": str(probe["probe_id"]),
            "capability": str(probe["canonical_capability"]),
            "output": value,
            "output_token_ids": tokens,
            "automatic_route": route,
            "functional_pass": evaluate_functional(value, probe["evaluator"]),
            "repetition_collapse_v1": repetition_collapse(value),
            "repetition_collapse_v2": repetition_collapse_v2(value),
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
        per[capability] = {"passes": passes, "observations": len(values), "v2_collapses": sum(bool(row["repetition_collapse_v2"]) for row in values), "wilson": wilson(passes, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10000, seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    absolute = protocol["absolute_screen"]
    v2_collapses = sum(bool(row["repetition_collapse_v2"]) for row in rows)
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= float(absolute["per_capability_functional_point_estimate_minimum"]) and value["wilson"]["lower_95"] >= float(absolute["per_capability_functional_wilson_lower_minimum"]) for value in per.values()),
        "critical_capabilities": all(per[name]["wilson"]["point"] >= float(absolute["critical_point_minimum"]) and per[name]["wilson"]["lower_95"] >= float(absolute["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_v2_repetition_collapses": v2_collapses == 0,
        "teacher_relative_noninferiority": relative["lower_95"] >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "frozen_parent": metadata["parent"]["mutated"] is False and metadata["parent"]["state_sha256_before"] == metadata["parent"]["state_sha256_after"],
        "teacher_absent_at_inference": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-v443-prompt-pointer-result/1",
        "status": "PASS_INITIAL_POINTER_SCREEN" if passed else "FAIL_POINTER_SUCCESSOR_CLOSED",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v1_diagnostic": sum(bool(row["repetition_collapse_v1"]) for row in rows),
        "repetition_collapses_v2": v2_collapses,
        "teacher_comparison": relative,
        "gates": gates,
        "passed": passed,
        "raw_outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "claim_boundary": "Single development-only sparse pointer successor; runtime, independent seeds, final quality, minimum information, and Phase 3 remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_V443_PROMPT_POINTER_PROTOCOL_V450.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--candidate-dir", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = root / args.protocol
    if args.command == "preflight":
        result = preflight(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, root / args.output_dir)
    else:
        result = evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
