"""Evidence-bound placement and hard-routed expert-bank Phase 3 successor."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import platform
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_sparse_router as sparse
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
from .capability_compiler_phase3_bpe_core import _json, _layercake_api, _model, _tokenizer
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_route_bridge import _base, _collate, BOS_ID, PAD_ID
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-resilience/1"
KNOWN_DOMAINS = ("chemistry", "civics", "mathematics", "python")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_HARD_ROUTED_EXPERT_BANK_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training", {}).get("device") != "cuda"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("phase4_status") != "LOCKED"
    ):
        raise Phase3Error("Phase 3 resilience governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 3 resilience binding changed: {relative}")
    return protocol, sha256_file(path)


def placement(root: Path, protocol_path: Path, selected_domains: Sequence[str]) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    phase1 = _json(root / protocol["phase1_certificate"])
    labeling = _json(root / protocol["labeling_certificate"])
    if phase1.get("status") != "PASS" or labeling.get("status") != "PASS_BOUNDED_LABELING_PHASE_COMPLETE":
        raise Phase3Error("teacher intake evidence is not qualified")
    selected = list(dict.fromkeys(str(value).strip().lower() for value in selected_domains if str(value).strip()))
    unsupported = sorted(set(selected) - set(KNOWN_DOMAINS))
    english = [
        {
            "capability": capability,
            "destination": "english_core",
            "acquisition_records": int(phase1["normalized_ir"]["selected_records_per_english_capability"]),
            "transfer_readiness": "READY_FOR_PHASE3_REALIZATION",
        }
        for capability in CAPABILITIES
    ]
    domains = []
    for domain in KNOWN_DOMAINS:
        requested = domain in selected
        domains.append(
            {
                "domain": domain,
                "destination": f"domain_cake:{domain}",
                "requested": requested,
                "diagnosis_scope": "BOUNDED_V89_ONTOLOGY_PASS",
                "reference_records": 100,
                "acquisition_records": 0,
                "transfer_readiness": "BLOCKED_NO_DOMAIN_ACQUISITION_PAYLOAD" if requested else "NOT_SELECTED",
            }
        )
    status = "FAIL_CLOSED_UNSUPPORTED_DOMAIN" if unsupported else (
        "BLOCKED_SELECTED_DOMAINS_LACK_ACQUISITION_PAYLOAD" if selected else "PASS_ENGLISH_PLACEMENT_ONLY"
    )
    result = {
        "format": "abi-capability-compiler-phase3-capability-placement/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "teacher": {
            "model": phase1["source"]["model"],
            "revision": phase1["source"]["revision"],
            "source_manifest_sha256": phase1["source"]["source_manifest_sha256"],
            "diagnosis_method": "frozen_teacher_probe_outputs_plus_bounded_record_labeling",
            "exhaustive_weight_knowledge_discovery": False,
        },
        "english_core": english,
        "domains": domains,
        "unsupported_selected_domains": unsupported,
        "unknown_or_ambiguous_destination": "quarantine",
        "layercake_plan": {
            "english": "one routed English-core expert bank candidate",
            "selected_domains": "one immutable domain cake per selected domain after a separately certified acquisition payload exists",
            "dynamic_activation": "router selects only the required English capability or domain cake",
        },
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lines = str(row["normalized_acquisition_prompt"]).splitlines()
        body = "\n".join(lines[1:]).strip()
        source_ids = [tokenizer.lexeme_to_id[piece] for piece in tokenizer.split(body)]
        target = [tokenizer.lexeme_to_id[piece] for piece in tokenizer.split(str(row["normalized_output"]))] + [2]
        if not source_ids or len(source_ids) > int(protocol["architecture"]["maximum_source_lexemes"]):
            raise Phase3Error("expert-bank source exceeds host bound")
        if len(target) > int(protocol["architecture"]["maximum_target_actions"]):
            raise Phase3Error("expert-bank target exceeds host bound")
        grouped[str(row["capability"])].append(
            {"record_id": str(row["ir_record_id"]), "capability": str(row["capability"]), "source_ids": source_ids, "target_actions": target}
        )
    if set(grouped) != set(CAPABILITIES) or any(len(grouped[value]) != 500 for value in CAPABILITIES):
        raise Phase3Error("expert-bank capability inventory changed")
    return grouped


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer = _base(root, protocol, torch.device("cpu"))
    grouped = _examples(root, protocol, tokenizer)
    per_expert = model.parameter_count()
    if per_expert != int(protocol["training"]["parameters_per_expert"]):
        raise Phase3Error("expert parameter count changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "experts": len(grouped),
        "records": sum(len(value) for value in grouped.values()),
        "records_per_expert": {key: len(value) for key, value in sorted(grouped.items())},
        "parameters_per_expert": per_expert,
        "stored_generator_parameters": per_expert * len(grouped),
        "active_generator_parameters": per_expert,
        "active_router_parameters": int(protocol["router"]["parameters"]),
        "maximum_source_actions": max(len(row["source_ids"]) for values in grouped.values() for row in values),
        "maximum_target_actions": max(len(row["target_actions"]) for values in grouped.values() for row in values),
        "teacher_outputs_added": 0,
        "source_parameters_copied": 0,
        "layercake_host_changed": False,
        "final_test_accessed": False,
    }


def _save_model(model: torch.nn.Module, path: Path) -> None:
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(path))


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("expert-bank output exists or CUDA unavailable")
    device = torch.device("cuda")
    _, tokenizer = _base(root, protocol, torch.device("cpu"))
    grouped = _examples(root, protocol, tokenizer)
    config = protocol["training"]
    seed = int(config["seed"])
    set_determinism(seed)
    output.mkdir(parents=True)
    experts_dir = output / "experts"
    experts_dir.mkdir()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    expert_rows = []
    for expert_index, capability in enumerate(CAPABILITIES):
        model, _ = _base(root, protocol, device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        expert_examples = grouped[capability]
        sampler = random.Random(seed + expert_index)
        successful = skipped = 0
        sequence = hashlib.sha256()
        curves = []
        expert_started = time.perf_counter()
        while successful < int(config["steps_per_expert"]):
            batch = [expert_examples[sampler.randrange(len(expert_examples))] for _ in range(int(config["batch_size"]))]
            source, targets = _collate(batch, device)
            previous = torch.full_like(targets, PAD_ID)
            previous[:, 0] = BOS_ID
            if targets.shape[1] > 1:
                previous[:, 1:] = torch.where(targets[:, :-1].ge(0), targets[:, :-1], torch.full_like(targets[:, :-1], PAD_ID))
            while True:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    log_probs = model.action_log_probs(source, previous)["log_probs"]
                    loss = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < before:
                    skipped += 1
                    continue
                break
            successful += 1
            for row in batch:
                sequence.update(row["record_id"].encode("utf-8") + b"\n")
            peak_rss = max(peak_rss, process.memory_info().rss)
            if successful == 1 or successful % int(config["curve_interval"]) == 0:
                curves.append({"step": successful, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - expert_started})
        checkpoint = experts_dir / f"{capability}.safetensors"
        _save_model(model, checkpoint)
        row = {
            "capability": capability,
            "checkpoint_path": f"experts/{capability}.safetensors",
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "steps": successful,
            "skipped_amp_steps": skipped,
            "record_sequence_sha256": sequence.hexdigest(),
            "wall_seconds": time.perf_counter() - expert_started,
            "curves": curves,
        }
        expert_rows.append(row)
        print(json.dumps({"trained_expert": capability, "steps": successful, "loss": curves[-1]["loss"], "wall_seconds": row["wall_seconds"]}), flush=True)
        del optimizer, scaler, model
        torch.cuda.empty_cache()
    router_source = (root / protocol["router"]["checkpoint_path"]).resolve()
    router_target = output / "router.safetensors"
    router_target.write_bytes(router_source.read_bytes())
    tokenizer_path = output / "tokenizer.json"
    _write_json(tokenizer_path, tokenizer.canonical_dict())
    config_path = output / "model_config.json"
    _write_json(config_path, {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size})
    metadata = {
        "format": "abi-capability-compiler-phase3-hard-routed-expert-bank/1",
        "status": "TRAINED_INITIAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "experts": expert_rows,
        "expert_manifest_sha256": hashlib.sha256(canonical_json_bytes(expert_rows)).hexdigest(),
        "router": {"sha256": sha256_file(router_target), "parameters": int(protocol["router"]["parameters"])},
        "tokenizer": {"sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size},
        "model_config_sha256": sha256_file(config_path),
        "parameters": {
            "per_expert": int(config["parameters_per_expert"]),
            "stored_generator": int(config["parameters_per_expert"]) * len(CAPABILITIES),
            "active_generator": int(config["parameters_per_expert"]),
            "active_router": int(protocol["router"]["parameters"]),
        },
        "training": {"steps_per_expert": int(config["steps_per_expert"]), "batch_size": int(config["batch_size"]), "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()},
        "imported_information": {"records": 7000, "teacher_outputs_added": 0, "stored_logits": 0, "stored_activations": 0, "source_parameters_copied": 0},
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "promotion_eligible": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


def _load_expert(root: Path, protocol: Mapping[str, Any], candidate: Path, capability: str, device: torch.device):
    _, model_type, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    model = _model(protocol, tokenizer, model_type)
    model.load_state_dict(load_file(str(candidate / "experts" / f"{capability}.safetensors"), device=str(device)), strict=True)
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("expert-bank evaluation output exists")
    metadata = _json(candidate / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha:
        raise Phase3Error("expert-bank candidate protocol changed")
    for row in metadata["experts"]:
        if sha256_file(candidate / row["checkpoint_path"]) != row["checkpoint_sha256"]:
            raise Phase3Error("expert-bank checkpoint changed")
    router_protocol = _json(root / protocol["router"]["protocol_path"])
    router, router_tokenizer = sparse._load(root, router_protocol, (root / protocol["router"]["candidate_dir"]).resolve())
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    by_capability: dict[str, list[tuple[int, Mapping[str, Any], str]]] = defaultdict(list)
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        route, _ = sparse._route(router, router_tokenizer, router_protocol, prompt)
        by_capability[route].append((index, probe, _semantic_segments(prompt)[-1]))
    rows: list[dict[str, Any] | None] = [None] * len(probes)
    device = torch.device("cuda")
    started = time.perf_counter()
    load_seconds = 0.0
    for capability in CAPABILITIES:
        load_started = time.perf_counter()
        model, _ = _load_expert(root, protocol, candidate, capability, device)
        load_seconds += time.perf_counter() - load_started
        for index, probe, body in by_capability[capability]:
            error = None
            try:
                value = model.generate_bytes(body, maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8", errors="strict")
            except Exception as exc:
                value = ""
                error = f"{type(exc).__name__}: {exc}"
            rows[index] = {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "predicted_route": capability,
                "route_correct": capability == str(probe["canonical_capability"]),
                "active_expert": capability,
                "output": value,
                "generation_error": error,
                "functional_pass": evaluate_functional(value, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(value),
            }
        print(json.dumps({"evaluated_expert": capability, "observations": len(by_capability[capability])}), flush=True)
        del model
        torch.cuda.empty_cache()
    completed = [row for row in rows if row is not None]
    if len(completed) != 1400:
        raise Phase3Error("expert-bank evaluation incomplete")
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in completed))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in completed if row["capability"] == capability]
        passes = sum(row["functional_pass"] for row in values)
        per[capability] = {"passes": passes, "observations": len(values), "collapses": sum(row["repetition_collapse"] for row in values), "wilson": wilson(passes, len(values))}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    probe_map = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in completed]
    comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    gate = protocol["absolute_screen"]
    collapses = sum(row["repetition_collapse"] for row in completed)
    errors = sum(row["generation_error"] is not None for row in completed)
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= gate["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= gate["per_capability_functional_wilson_lower_minimum"] for value in per.values()),
        "critical_capabilities": all(per[value]["wilson"]["point"] >= gate["critical_point_minimum"] and per[value]["wilson"]["lower_95"] >= gate["critical_wilson_lower_minimum"] for value in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapses": collapses == 0,
        "zero_generation_errors": errors == 0,
        "router_accuracy": sum(row["route_correct"] for row in completed) == len(completed),
        "teacher_relative_noninferiority": comparison["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"],
    }
    passed = all(gates.values())
    decision = {
        "format": "abi-capability-compiler-phase3-hard-routed-expert-bank-decision/1",
        "status": "PASS_INITIAL_SCREEN_REPLICATION_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_EXPERT_BANK_CLOSED",
        "protocol_sha256": protocol_sha,
        "expert_manifest_sha256": metadata["expert_manifest_sha256"],
        "functional_passes": sum(row["functional_pass"] for row in completed),
        "observations": len(completed),
        "per_capability": per,
        "repetition_collapses": collapses,
        "generation_errors": errors,
        "route_correct": sum(row["route_correct"] for row in completed),
        "teacher_comparison": comparison,
        "gates": gates,
        "initial_screen_pass": passed,
        "promotion_eligible": False,
        "outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "expert_load_seconds": load_seconds,
        "stored_generator_parameters": metadata["parameters"]["stored_generator"],
        "active_generator_parameters": metadata["parameters"]["active_generator"],
        "active_router_parameters": metadata["parameters"]["active_router"],
        "teacher_present_at_inference": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "next_step": "Preregister paired seeds and same-candidate LayerCake host/runtime certification." if passed else "Preserve failure; do not run remaining seeds or nearby expert-bank variants.",
    }
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    _write_json(output / "decision.json", decision)
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("placement", "inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_PROTOCOL_V52.json")
    parser.add_argument("--selected-domain", action="append", default=[])
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_resilience/development_v52/E0-seed240052")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_resilience/evaluation_v52/E0-seed240052")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.command == "placement":
        result = placement(root, protocol, args.selected_domain)
    elif args.command == "inventory":
        result = inventory(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, (root / args.candidate_dir).resolve())
    else:
        result = evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
