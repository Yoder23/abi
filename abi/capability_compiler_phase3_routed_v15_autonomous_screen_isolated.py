"""Import-isolated locked autonomous screen for the assembled routed-v15 artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable

import psutil
from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, sha256_file
from .capability_compiler_phase2_teacher import development_probes
FORMAT = "abi-capability-compiler-phase3-routed-v15-autonomous-screen-isolated/1"
ROUTES = ("generic", "abstention", "conversation")


class Phase3Error(RuntimeError):
    pass


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        raise Phase3Error(f"autonomous-screen evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def expected_route(capability: str) -> str:
    if capability not in CAPABILITIES:
        raise Phase3Error("unknown capability in routed-v15 screen")
    return capability if capability in ROUTES[1:] else "generic"


def semantic_segments(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return [text.strip()]
    first = lines[0].strip()
    remainder = "\n".join(lines[1:]).strip()
    return [part for part in (first, remainder) if part]


def controlled_prompt(capability: str, prompt: str) -> str:
    body = semantic_segments(prompt)[-1].strip()
    if not body:
        raise Phase3Error("empty semantic prompt body")
    return f"Capability route: {capability}\n{body}"


def wilson(successes: int, observations: int, z: float = 1.959963984540054) -> dict[str, float]:
    if observations <= 0 or not 0 <= successes <= observations:
        raise Phase3Error("invalid Wilson inputs")
    point = successes / observations
    denominator = 1.0 + z * z / observations
    center = (point + z * z / (2.0 * observations)) / denominator
    half = z * math.sqrt(point * (1.0 - point) / observations + z * z / (4.0 * observations**2)) / denominator
    return {"point": point, "lower_95": center - half, "upper_95": center + half}


def paired_stratified_bootstrap(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    strata = {
        capability: [int(row["candidate_pass"]) - int(row["teacher_pass"]) for row in rows if row["capability"] == capability]
        for capability in CAPABILITIES
    }
    if any(len(values) != 100 for values in strata.values()):
        raise Phase3Error("paired bootstrap depth changed")
    observed = sum(sum(values) for values in strata.values()) / len(rows)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        total = sum(sum(values[rng.randrange(len(values))] for _ in values) for values in strata.values())
        draws.append(total / len(rows))
    draws.sort()
    return {
        "candidate_minus_teacher": observed,
        "lower_95": draws[int(0.025 * replicates)],
        "upper_95": draws[min(replicates - 1, int(0.975 * replicates))],
        "replicates": replicates,
        "seed": seed,
        "method": "capability_stratified_paired_percentile_bootstrap",
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _validate_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_IMPORT_ISOLATED_LOCKED_1400_PROMPT_AUTONOMOUS_SCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("source_model_access") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("prompt_count") != 1400
        or protocol.get("prompts_per_capability") != 100
    ):
        raise Phase3Error("isolated autonomous-screen governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"isolated autonomous-screen binding changed: {name}")
    return protocol, sha256_file(path)


def _load_artifact(root: Path, protocol: dict[str, Any]):
    if "transformers" in sys.modules:
        raise Phase3Error("source-model runtime imported before autonomous screen")
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    manifest, config = _json(artifact / "manifest.json"), _json(artifact / "config.json")
    if (
        manifest.get("manifest_sha256") != protocol["artifact"]["manifest_self_sha256"]
        or manifest.get("artifact_promoted") is not False
        or manifest.get("source", {}).get("source_blocks_in_artifact") != 0
        or manifest.get("source", {}).get("teacher_present_in_artifact") is not False
        or config.get("source_transformer_blocks") != 0
        or config.get("teacher_required_at_inference") is not False
        or config.get("abi_sha256") != protocol["artifact"]["abi_sha256"]
    ):
        raise Phase3Error("routed-v15 artifact boundary changed")
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = RoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    state = load_file(str(artifact / "model.safetensors"), device="cuda")
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Phase3Error("routed-v15 strict artifact load failed")
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != protocol["artifact"]["parameters"]:
        raise Phase3Error("routed-v15 parameter count changed")
    if "transformers" in sys.modules:
        raise Phase3Error("source-model runtime imported during artifact load")
    return model, tokenizer, manifest


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = _validate_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available() or "transformers" in sys.modules:
        raise Phase3Error("screen output exists, CUDA unavailable, or source runtime imported")
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    counts: dict[str, int] = defaultdict(int)
    for probe in probes:
        counts[str(probe["canonical_capability"])] += 1
    if len(probes) != 1400 or set(counts) != set(CAPABILITIES) or any(value != 100 for value in counts.values()):
        raise Phase3Error("development depth or stratification changed")
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines())}
    if set(teacher) != {str(probe["probe_id"]) for probe in probes}:
        raise Phase3Error("teacher development reference join changed")
    model, tokenizer, manifest = _load_artifact(root, protocol)
    output.mkdir(parents=True)
    raw_path = output / "development_outputs.jsonl"
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    with raw_path.open("xb") as raw:
        for index, probe in enumerate(probes):
            capability = str(probe["canonical_capability"])
            rendered = controlled_prompt(capability, str(probe["prompt"]))
            expected = expected_route(capability)
            error, value, predicted = None, "", "ERROR"
            generated_actions = source_actions = 0
            try:
                source_ids, source_lexemes = tokenizer.encode_source(rendered)
                source_actions = len(source_ids)
                if source_actions > int(protocol["limits"]["maximum_source_actions"]):
                    raise Phase3Error("development prompt exceeds source bound")
                predicted = model.route_names[model._select_route(torch.tensor([source_ids], dtype=torch.long, device="cuda"))]
                state = model.prefill_ids(source_ids, source_lexemes)
                maximum = min(int(probe["max_new_tokens"]), int(protocol["limits"]["maximum_target_actions"]))
                while not state.complete and len(state.generated_actions) < maximum:
                    model.decode_step(state)
                generated_actions = len(state.generated_actions)
                value = tokenizer.decode_actions(state.generated_actions, state.source_lexemes).decode("utf-8", errors="strict")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "probe_id": str(probe["probe_id"]), "capability": capability,
                "expected_route": expected, "predicted_route": predicted,
                "route_correct": predicted == expected, "source_actions": source_actions,
                "generated_actions": generated_actions, "output": value,
                "generation_error": error, "functional_pass": evaluate_functional(value, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(value),
            }
            rows.append(row); raw.write(canonical_json_bytes(row)); raw.flush()
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 25 == 0:
                print(json.dumps({"evaluated": index + 1, "passes": sum(r["functional_pass"] for r in rows), "collapses": sum(r["repetition_collapse"] for r in rows)}), flush=True)
    elapsed = time.perf_counter() - started
    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(bool(row["functional_pass"]) for row in values)
        per_capability[capability] = {"passes": passes, "observations": len(values), "collapses": sum(bool(row["repetition_collapse"]) for row in values), "wilson": wilson(passes, len(values))}
    probe_map = {str(probe["probe_id"]): probe for probe in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]
    comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    gate = protocol["absolute_screen"]
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= gate["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= gate["per_capability_functional_wilson_lower_minimum"] for value in per_capability.values()),
        "critical_capabilities": all(per_capability[name]["wilson"]["point"] >= gate["critical_point_minimum"] and per_capability[name]["wilson"]["lower_95"] >= gate["critical_wilson_lower_minimum"] for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows) <= gate["repetition_collapse_count_maximum"],
        "zero_generation_errors": sum(row["generation_error"] is not None for row in rows) <= gate["generation_error_count_maximum"],
        "router_accuracy": sum(bool(row["route_correct"]) for row in rows) / len(rows) >= gate["router_accuracy_minimum"],
        "teacher_relative_noninferiority": comparison["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"],
    }
    passed = all(gates.values())
    decision = {
        "format": FORMAT, "status": "PASS_AUTONOMOUS_ENGLISH_DEVELOPMENT_SCREEN" if passed else "FAIL_AUTONOMOUS_ENGLISH_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "artifact": {"model_sha256": protocol["artifact"]["model_sha256"], "manifest_sha256": protocol["artifact"]["manifest_sha256"], "parameters": protocol["artifact"]["parameters"], "source_blocks": manifest["source"]["source_blocks_in_artifact"]},
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows), "observations": len(rows),
        "per_capability": per_capability, "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows), "route_correct": sum(bool(row["route_correct"]) for row in rows),
        "teacher_comparison": comparison, "gates": gates, "passed": passed,
        "outputs_sha256": sha256_file(raw_path), "evaluation_wall_seconds": elapsed, "outputs_per_second": len(rows) / elapsed,
        "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        "teacher_present_at_inference": False, "source_model_loaded": False, "transformers_imported": "transformers" in sys.modules,
        "artifact_promoted": False, "phase3_certified": False, "final_test_accessed": False,
        "claim_boundary": "Development-only autonomous quality result for the exact routed-v15 artifact; no final-test, runtime-dominance, Phase 3 certification, or ABI-superiority claim.",
    }
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    _write_immutable(output / "decision.json", json.dumps(decision, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_AUTONOMOUS_SCREEN_PROTOCOL_V329.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v15/autonomous_screen_v330")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
