"""Permanently non-promotional oracle-fit capacity diagnostic for Phase 3."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import TOKENIZER_FILES, Phase3Error, _BalancedSampler, _batch, _state_hash, _write_immutable
from .capability_compiler_phase3_sequence_bridge import _generate
from .capability_compiler_phase3_shared_output import _is_trainable, load_candidate, load_protocol as load_v11_protocol


FORMAT = "abi-capability-compiler-phase3-oracle-fit-capacity/1"
EXPECTED_TRAINABLE_PARAMETERS = 1_057_798


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NONPROMOTIONAL_CAPACITY_DIAGNOSTIC"
        or protocol.get("development_contaminated") is not True or protocol.get("promotion_eligible") is not False
        or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("training", {}).get("steps") != 1400
        or protocol.get("training", {}).get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS
    ): raise Phase3Error("oracle-fit governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"oracle-fit binding changed: {relative}")
    return protocol, sha256_file(path)


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    teachers = [json.loads(line) for line in (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines() if line]
    routes = {cap: int(route) for route, caps in protocol["capability_routes"].items() for cap in caps}
    examples = []
    for teacher in teachers:
        probe_id = str(teacher["probe_id"]); probe = probe_by_id[probe_id]; capability = str(teacher["capability"])
        prompt_ids = [int(v) for v in tokenizer.encode(str(probe["prompt"]).rstrip() + "\n", add_special_tokens=False)]
        response_ids = [int(v) for v in tokenizer.encode(str(teacher["output"]), add_special_tokens=False)] + [int(tokenizer.eos_token_id)]
        ids = prompt_ids + response_ids
        if len(ids) > int(protocol["training"]["max_tokens"]): raise Phase3Error(f"oracle example exceeds context: {probe_id}")
        examples.append({
            "record_id": probe_id, "capability": capability, "route": routes[capability], "input_ids": ids,
            "labels": [-100] * len(prompt_ids) + response_ids, "prompt_tokens": len(prompt_ids), "response_tokens": len(response_ids),
        })
    if len(examples) != 1400 or len({v["record_id"] for v in examples}) != 1400: raise Phase3Error("oracle development examples changed")
    return examples


def _load(root: Path, protocol: Mapping[str, Any], device: torch.device):
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve())
    model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=(root / protocol["starting_candidate"]).resolve(), device=device)
    return model, tokenizer, v11


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path); model, tokenizer, _ = _load(root, protocol, torch.device("cpu")); examples = _examples(root, protocol, tokenizer)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(_is_trainable(name))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS: raise Phase3Error("oracle trainable count changed")
    return {"status": "PASS", "protocol_sha256": protocol_sha, "examples": len(examples), "trainable_parameters": trainable, "development_contaminated": True, "promotion_eligible": False, "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists() or not torch.cuda.is_available(): raise Phase3Error("oracle output exists or GPU unavailable")
    cfg = protocol["training"]; seed = int(cfg["seed"]); set_determinism(seed); device = torch.device("cuda")
    model, tokenizer, v11 = _load(root, protocol, device); examples = _examples(root, protocol, tokenizer); sampler = _BalancedSampler(examples, seed)
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_trainable(name))
        if parameter.requires_grad: trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1); scaler = torch.amp.GradScaler("cuda", enabled=True)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    successful = skipped = 0; sampled = Counter(); sampled_hash = hashlib.sha256(); curves = []; model.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        while True:
            ids, labels, attention, prompt_lengths, routes = _batch(selected, int(tokenizer.eos_token_id), device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
                language_loss = F.cross_entropy(result["logits"][:, :-1].float().reshape(-1, result["logits"].shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
                classifier_loss = F.cross_entropy(result["task_logits"].float(), routes); loss = language_loss + float(cfg["classifier_loss_weight"]) * classifier_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(trainable, 1.0); scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < scale_before: skipped += 1; continue
            break
        successful += 1
        for row in selected: sampled[row["capability"]] += 1; sampled_hash.update(str(row["record_id"]).encode("ascii") + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curves.append({"step": successful, "language_loss": float(language_loss.detach()), "classifier_loss": float(classifier_loss.detach()), "wall_seconds": time.perf_counter() - started}); print(json.dumps(curves[-1]), flush=True)
    model.eval(); after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}; changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(not _is_trainable(name) for name in changed): raise Phase3Error("oracle changed a frozen tensor")
    frozen_before = {n: v for n, v in before.items() if not _is_trainable(n)}; frozen_after = {n: v for n, v in after.items() if not _is_trainable(n)}
    if _state_hash(frozen_before) != _state_hash(frozen_after): raise Phase3Error("oracle frozen host changed")
    output_dir.mkdir(parents=True); checkpoint = output_dir / "model.safetensors"; save_file(after, str(checkpoint)); parent = (root / v11["host"]["parent_path"]).resolve()
    for name in TOKENIZER_FILES: shutil.copyfile(parent / name, output_dir / name)
    wall = time.perf_counter() - started
    metadata = {
        "format": "abi-capability-compiler-phase3-oracle-fit-candidate/1", "status": "TRAINED_DIAGNOSTIC_ONLY_NEVER_PROMOTABLE", "protocol_sha256": protocol_sha, "seed": seed,
        "development_contaminated": True, "promotion_eligible": False, "final_test_accessed": False, "teacher_present_at_inference": False,
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "starting_checkpoint_sha256": protocol["starting_checkpoint_sha256"],
        "training": {"steps": successful, "batch_size": int(cfg["batch_size"]), "wall_seconds": wall, "skipped_amp_steps": skipped, "successful_record_sequence_sha256": sampled_hash.hexdigest(), "sampled_records_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves},
        "isolation": {"changed_tensors": changed, "all_changes_confined_to_registered_bridge": True, "frozen_state_sha256_before": _state_hash(frozen_before), "frozen_state_sha256_after": _state_hash(frozen_after)},
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    metadata["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"); return metadata


def evaluate(root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path); metadata = _json(candidate_dir / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate_dir / "model.safetensors") != metadata["checkpoint"]["sha256"] or output_dir.exists(): raise Phase3Error("oracle evaluation identity or immutability failure")
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve()); model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=candidate_dir, device=torch.device("cuda")); probes = development_probes((root / protocol["development_catalog"]).resolve()); rows = []; started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), torch.device("cuda")); rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "output": output, "output_token_ids": token_ids, "automatic_route": route, "functional_pass": evaluate_functional(output, probe["evaluator"]), "repetition_collapse": repetition_collapse(output)})
        if (index + 1) % 100 == 0: print(json.dumps({"evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True); outputs = output_dir / "development_outputs.jsonl"; outputs.write_bytes(b"".join(canonical_json_bytes(row) for row in rows)); grouped = {cap: [r for r in rows if r["capability"] == cap] for cap in CAPABILITIES}
    receipt = {"format": "abi-capability-compiler-phase3-oracle-fit-evaluation/1", "status": "PASS_EXECUTION_DIAGNOSTIC_ONLY", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "development_contaminated": True, "promotion_eligible": False, "observations": len(rows), "functional_passes": sum(bool(r["functional_pass"]) for r in rows), "repetition_collapses": sum(bool(r["repetition_collapse"]) for r in rows), "per_capability": {cap: {"passes": sum(bool(r["functional_pass"]) for r in values), "collapses": sum(bool(r["repetition_collapse"]) for r in values), "observations": len(values)} for cap, values in grouped.items()}, "outputs_path": outputs.relative_to(root).as_posix(), "outputs_sha256": sha256_file(outputs), "wall_seconds": time.perf_counter() - started, "final_test_accessed": False}
    _write_immutable(output_dir / "receipt.json", canonical_json_bytes(receipt)); return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ORACLE_FIT_PROTOCOL_V19.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); tr = sub.add_parser("train"); tr.add_argument("--output-dir", required=True); ev = sub.add_parser("evaluate"); ev.add_argument("--candidate-dir", required=True); ev.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve()
    if args.command == "preflight": result = preflight(root, protocol)
    elif args.command == "train": result = train(root, protocol, (root / args.output_dir).resolve())
    else: result = evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
