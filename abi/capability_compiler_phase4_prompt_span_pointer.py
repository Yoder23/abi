"""Evaluator-blind prompt-span pointer feasibility for the frozen Phase 4 host."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase4_abi_lineage as lineage
from . import capability_compiler_phase4_capability_isolated_adaptation as adapted
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-prompt-span-pointer/1"
_BRACKETED = re.compile(r"\[[^\[\]\r\n]{1,128}\]")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return rows


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EVALUATOR_BLIND_PROMPT_SPAN_POINTER_FEASIBILITY"
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_construction_authorized") is not False
        or protocol.get("promotion_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("prompt-span pointer governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"prompt-span pointer binding changed: {relative}")
    lineage_protocol, _ = lineage.load_protocol(root, root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def extract_segments(prompt: str) -> tuple[str, ...]:
    matches = list(_BRACKETED.finditer(prompt))
    segments = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        segment = prompt[match.start():end].strip().rstrip(";.").strip()
        segments.append(segment)
    return tuple(segments)


def render_segments(segments: Iterable[str]) -> str:
    return "; ".join(segment.rstrip(";.").strip() for segment in segments) + "."


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    rows = _jsonl(root / protocol["suite"])
    segment_sets = [extract_segments(str(row["normalized_generation_prompt"])) for row in rows]
    gates = {
        "expected_depth": len(rows) == int(protocol["expected_records"]),
        "exactly_three_segments": all(len(segments) == 3 for segments in segment_sets),
        "segments_unique_per_prompt": all(len(set(segments)) == 3 for segments in segment_sets),
        "six_permutations_per_prompt": all(len(set(itertools.permutations(segments))) == 6 for segments in segment_sets),
        "literal_round_trip": all(all(segment in row["normalized_generation_prompt"] for segment in segments) for row, segments in zip(rows, segment_sets)),
    }
    return {
        "status": "PASS_PROMPT_SPAN_POINTER_PREFLIGHT" if all(gates.values()) else "FAIL_PROMPT_SPAN_POINTER_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "records": len(rows),
        "candidate_permutations_per_record": 6,
        "planned_scoring_forward_passes": len(rows) * 6,
        "gates": gates,
        "model_inference_performed": False,
        "evaluator_used_for_selection": False,
        "training_performed": False,
        "candidate_constructed": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def _score_candidate(model: Any, tokenizer: Any, prompt: str, candidate: str, device: torch.device) -> tuple[float, int]:
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    candidate_ids = [int(value) for value in tokenizer.encode(candidate, add_special_tokens=False)]
    if not prompt_ids or not candidate_ids:
        raise Phase3Error("pointer scoring received an empty token sequence")
    ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    adapted._set_routes(model, torch.tensor([CAPABILITIES.index("coherence")], dtype=torch.long, device=device))
    task = torch.tensor([lineage.CAPABILITY_TO_ROUTE["coherence"]], dtype=torch.long, device=device)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=task, use_cache=False)
    logits = result["logits"][0, len(prompt_ids) - 1:-1].float()
    targets = ids[0, len(prompt_ids):]
    if logits.shape[0] != targets.shape[0]:
        raise Phase3Error("pointer candidate score alignment changed")
    log_probabilities = F.log_softmax(logits, dim=-1)
    score = log_probabilities.gather(1, targets[:, None]).sum().item()
    return float(score), len(candidate_ids)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable pointer output exists or CUDA unavailable")
    rows = _jsonl(root / protocol["suite"])
    device = torch.device("cuda")
    run_spec = protocol["runs"][0]
    run_dir = root / run_spec["lineage_dir"]
    model, tokenizer, _, _, _ = adapted._load_components(root, protocol, lineage_protocol, run_spec, device)
    residual = adapted.CapabilityIsolatedResidual().to(device)
    residual.load_state_dict(load_file(str(root / protocol["adapted_checkpoint"]), device="cuda"), strict=True)
    residual.eval()
    handles = adapted._attach(model, residual)
    evidence = []
    started = time.perf_counter()
    candidate_tokens_scored = 0
    try:
        for row in sorted(rows, key=lambda item: str(item["ir_record_id"])):
            prompt = str(row["normalized_generation_prompt"])
            segments = extract_segments(prompt)
            if len(segments) != 3 or len(set(segments)) != 3:
                raise Phase3Error("pointer prompt segment inventory changed")
            candidates = [render_segments(permutation) for permutation in itertools.permutations(segments)]
            scores = []
            for candidate in candidates:
                score, tokens = _score_candidate(model, tokenizer, prompt, candidate, device)
                scores.append(score)
                candidate_tokens_scored += tokens
            selected_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            selected = candidates[selected_index]
            passed = evaluate_functional(selected, row["functional_evaluator"])
            evidence.append({
                "record_id": row["ir_record_id"],
                "namespace": row["namespace"],
                "family": row["family"],
                "selected_permutation_index": selected_index,
                "permutation_log_probability_sums": scores,
                "functional_pass_v1": passed,
                "repetition_collapse_v2": repetition_collapse_v2(selected),
                "output": selected,
            })
    finally:
        for handle in handles:
            handle.remove()
    wall = time.perf_counter() - started
    passes = sum(row["functional_pass_v1"] for row in evidence)
    interval = wilson(passes, len(evidence))
    by_namespace = {name: {"passes": sum(row["functional_pass_v1"] for row in evidence if row["namespace"] == name), "observations": sum(row["namespace"] == name for row in evidence)} for name in sorted({row["namespace"] for row in evidence})}
    by_family = {str(family): {"passes": sum(row["functional_pass_v1"] for row in evidence if row["family"] == family), "observations": sum(row["family"] == family for row in evidence)} for family in range(4)}
    gates = {
        "point": interval["point"] >= float(protocol["thresholds"]["point"]),
        "lower_95": interval["lower_95"] >= float(protocol["thresholds"]["lower_95"]),
        "per_namespace_point": min(value["passes"] / value["observations"] for value in by_namespace.values()) >= float(protocol["thresholds"]["per_stratum_point"]),
        "per_family_point": min(value["passes"] / value["observations"] for value in by_family.values()) >= float(protocol["thresholds"]["per_stratum_point"]),
        "zero_collapse": not any(row["repetition_collapse_v2"] for row in evidence),
        "evaluator_blind_selection": True,
        "no_training": True,
        "no_candidate": True,
        "final_test_not_accessed": True,
    }
    raw = output.parent / "outputs.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in evidence))
    result = {
        "format": "abi-capability-compiler-phase4-prompt-span-pointer-result/1",
        "status": "PASS_PROMPT_SPAN_POINTER_HOST_REQUIREMENT_FEASIBILITY" if all(gates.values()) else "FAIL_PROMPT_SPAN_POINTER_FEASIBILITY",
        "protocol_sha256": protocol_sha,
        "observations": len(evidence),
        "functional_passes": passes,
        "wilson": interval,
        "by_namespace": by_namespace,
        "by_family": by_family,
        "scoring": {"forward_passes": len(evidence) * 6, "candidate_tokens_scored": candidate_tokens_scored, "gpu_wall_seconds": wall, "records_per_second": len(evidence) / wall},
        "gates": gates,
        "outputs_sha256": sha256_file(raw),
        "evaluator_used_for_selection": False,
        "model_inference_performed": True,
        "training_performed": False,
        "candidate_constructed": False,
        "promotion_authorized": False,
        "final_test_accessed": False,
        "interpretation": "A pass identifies a feasible generic host interface for exact prompt-span realization. It is not ABI acquisition, a LayerCake implementation, or a promotable candidate.",
        "claim_boundary": "Host-requirement feasibility only; no ABI artifact, repaired LayerCake host, stable frontier, CPU runtime pass, matched baseline, final test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("run")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
