"""Raw-token identity hidden residual on the frozen V443 LayerCake parent."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_qualified_transition_control import _state_hash
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_sequence_bridge import _BalancedSampler, _batch, _examples
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-identity-residual/1"
RANK = 64
ROUTES = 10
PARAMETERS = 99_093


class ExactIdentityResidual(nn.Module):
    """Select raw prompt token embeddings and inject them before the tied head."""

    def __init__(self, width: int = 768, rank: int = RANK, routes: int = ROUTES):
        super().__init__()
        self.rank = int(rank)
        self.key = nn.Linear(width, rank, bias=False)
        self.query = nn.Linear(width, rank, bias=False)
        self.gate = nn.Linear(width, 1)
        self.route_bias = nn.Embedding(routes, 1)
        self.log_strength = nn.Embedding(routes, 1)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        nn.init.zeros_(self.route_bias.weight)
        nn.init.zeros_(self.log_strength.weight)

    def scores(self, hidden: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        return (self.query(hidden) @ self.key(prompt_embeddings).transpose(-1, -2)).float() / math.sqrt(self.rank)

    def copy_gate(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(hidden).squeeze(-1) + self.route_bias(routes.long()).squeeze(-1)).float()

    def adapt(self, hidden: torch.Tensor, prompt_embeddings: torch.Tensor, routes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = F.softmax(self.scores(hidden, prompt_embeddings), dim=-1)
        context = attention.to(prompt_embeddings.dtype) @ prompt_embeddings
        gate = self.copy_gate(hidden, routes)
        strength = self.log_strength(routes.long()).squeeze(-1).exp().to(hidden.dtype)
        return hidden + (gate * strength)[:, None].to(hidden.dtype) * context, attention


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_SINGLE_BOUNDED_SUCCESSOR" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("nearby_sweeps_authorized") is not False:
        raise Phase3Error("identity residual governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"identity residual binding changed: {name}")
    return protocol, sha256_file(path)


def _bridge(device: torch.device) -> ExactIdentityResidual:
    bridge = ExactIdentityResidual().to(device)
    if sum(value.numel() for value in bridge.parameters()) != PARAMETERS:
        raise Phase3Error("identity residual parameter count changed")
    return bridge


def _loss(model, bridge, result, ids, labels, prompt_lengths, routes):
    record_losses = []
    for index in range(ids.shape[0]):
        active = torch.nonzero(labels[index, 1:] >= 0, as_tuple=False).flatten()
        targets = labels[index, 1:].index_select(0, active).long()
        hidden = result["hidden"][index, :-1].index_select(0, active).detach()
        base_logits = result["logits"][index, :-1].index_select(0, active).detach().float()
        prompt_tokens = ids[index, : int(prompt_lengths[index])]
        prompt_embeddings = model.transformer.wte(prompt_tokens).detach()
        route_vector = routes[index].expand(hidden.shape[0])
        adapted, attention = bridge.adapt(hidden, prompt_embeddings, route_vector)
        language = F.cross_entropy(F.linear(adapted, model.output_weight).float(), targets)
        matches = prompt_tokens[None, :] == targets[:, None]
        copy_targets = matches.any(dim=-1) & (base_logits.argmax(dim=-1) != targets)
        positive = torch.nonzero(copy_targets, as_tuple=False).flatten()
        auxiliary = []
        if positive.numel():
            masked = attention.index_select(0, positive).float().log().masked_fill(~matches.index_select(0, positive), float("-inf"))
            auxiliary.append(-torch.logsumexp(masked, dim=-1).mean())
        gate = bridge.copy_gate(hidden, route_vector).clamp(1e-6, 1 - 1e-6)
        gate_terms = []
        if positive.numel():
            gate_terms.append(-gate.index_select(0, positive).log().mean())
        negative = torch.nonzero(~copy_targets, as_tuple=False).flatten()
        if negative.numel():
            gate_terms.append(-(1 - gate.index_select(0, negative)).log().mean())
        auxiliary.append(torch.stack(gate_terms).mean())
        record_losses.append(language + 0.25 * torch.stack(auxiliary).sum())
    return torch.stack(record_losses).mean()


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_v443(root, protocol, torch.device("cpu"))
    bridge = _bridge(torch.device("cpu"))
    return {"status": "PASS_PREFLIGHT", "protocol_sha256": protocol_sha, "frozen_parent_parameters": sum(value.numel() for value in model.parameters()), "new_trainable_parameters": sum(value.numel() for value in bridge.parameters()), "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable identity residual output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("identity residual CUDA unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    from .capability_compiler_phase2_common import set_determinism
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_v443(root, protocol, device)
    bridge = _bridge(device)
    rows = load_phase1_ir(root / protocol["phase1_ir"]["path"])
    examples = _examples(rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    sampler = _BalancedSampler(examples, seed)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(cfg["weight_decay"]))
    parent_before = _state_hash(model.state_dict())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    sampled = Counter()
    sequence_sha = hashlib.sha256()
    language_tokens = 0
    curves = []
    started = time.perf_counter()
    bridge.train()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.batch(int(cfg["batch_size"]))
        ids, labels, attention, prompt_lengths, routes = _batch(selected, int(tokenizer.eos_token_id), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = _loss(model, bridge, result, ids, labels, prompt_lengths, routes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(cfg["gradient_clip_norm"]))
        optimizer.step()
        for row in selected:
            sampled[str(row["capability"])] += 1
            sequence_sha.update(str(row["record_id"]).encode("ascii") + b"\n")
            language_tokens += int(row["response_tokens"])
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    bridge.eval()
    parent_after = _state_hash(model.state_dict())
    if parent_before != parent_after:
        raise Phase3Error("frozen V443 parent changed")
    output.mkdir(parents=True)
    checkpoint = output / "identity_residual.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()}, str(checkpoint), metadata={"format": FORMAT})
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT, "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL", "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "state_sha256_before": parent_before, "state_sha256_after": parent_after, "mutated": False},
        "bridge": {"rank": RANK, "routes": ROUTES, "parameters": PARAMETERS, "source_parameters_copied": 0},
        "training": {"device": "cuda", "seed": seed, "steps": int(cfg["steps"]), "batch_size": int(cfg["batch_size"]), "teacher_response_tokens_seen": language_tokens, "sampled_records_by_capability": dict(sorted(sampled.items())), "successful_record_sequence_sha256": sequence_sha.hexdigest(), "wall_seconds": wall, "active_parameter_seconds": PARAMETERS * wall, "peak_process_rss_bytes": int(peak_rss), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves},
        "teacher_present_at_training_or_inference": False, "source_blocks_retained": 0, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def _adapt_logits(model, bridge, hidden, prompt_embeddings, route):
    adapted, _ = bridge.adapt(hidden, prompt_embeddings, route.expand(hidden.shape[0]))
    return F.linear(adapted, model.output_weight).float()


@torch.inference_mode()
def _generate(model, bridge, tokenizer, prompt: str, maximum: int, device: torch.device):
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_embeddings = model.transformer.wte(ids[0])
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), use_cache=True)
    route = result["task_routes"]
    cache = result["past_key_values"]
    hidden = result["hidden"][:, -1]
    generated = []
    for _ in range(maximum):
        selected = _adapt_logits(model, bridge, hidden, prompt_embeddings, route).argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=route, past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        hidden = result["hidden"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated, int(route.item())


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable identity residual evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / metadata["checkpoint"]["path"]
    if metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("identity residual lineage changed")
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
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "output": value, "output_token_ids": tokens, "automatic_route": route, "functional_pass": evaluate_functional(value, probe["evaluator"]), "repetition_collapse_v2": repetition_collapse_v2(value)})
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] for row in values)
        per[capability] = {"passes": passes, "observations": len(values), "v2_collapses": sum(row["repetition_collapse_v2"] for row in values), "wilson": wilson(passes, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10000, seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    absolute = protocol["absolute_screen"]
    collapses = sum(row["repetition_collapse_v2"] for row in rows)
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= absolute["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= absolute["per_capability_functional_wilson_lower_minimum"] for value in per.values()),
        "critical_capabilities": all(per[name]["wilson"]["point"] >= absolute["critical_point_minimum"] and per[name]["wilson"]["lower_95"] >= absolute["critical_wilson_lower_minimum"] for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_v2_repetition_collapses": collapses == 0,
        "teacher_relative_noninferiority": relative["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"],
        "frozen_parent": metadata["parent"]["mutated"] is False and metadata["parent"]["state_sha256_before"] == metadata["parent"]["state_sha256_after"],
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {"format": "abi-capability-compiler-phase3-identity-residual-result/1", "status": "PASS_INITIAL_IDENTITY_RESIDUAL_SCREEN" if passed else "FAIL_IDENTITY_RESIDUAL_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes": sum(row["functional_pass"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses_v2": collapses, "teacher_comparison": relative, "gates": gates, "passed": passed, "raw_outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "claim_boundary": "Single development-only hidden identity residual; independent seeds, runtime, final quality, minimum information, and Phase 3 remain unproven."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_IDENTITY_RESIDUAL_PROTOCOL_V456.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train"); train_parser.add_argument("--output-dir", required=True)
    eval_parser = sub.add_parser("evaluate"); eval_parser.add_argument("--candidate-dir", required=True); eval_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
