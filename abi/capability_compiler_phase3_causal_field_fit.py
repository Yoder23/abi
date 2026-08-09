"""Read-only acquisition and held-out fit attribution for the sealed causal core."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_causal_field_core import _examples, _json, _load_candidate, _tokenizer, _layercake_types, BOS_ID, PAD_ID
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-causal-field-fit/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_ATTRIBUTION" or protocol.get("neural_training_authorized") is not False or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("causal-field fit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"causal-field fit binding changed: {relative}")
    return protocol, sha256_file(path)


def _packed(rows: list[Mapping[str, Any]], device: torch.device):
    width = max(len(row["source_ids"]) + len(row["target_actions"]) for row in rows)
    target_width = max(len(row["target_actions"]) for row in rows)
    inputs = torch.full((len(rows), width), PAD_ID, dtype=torch.long, device=device)
    positions = torch.zeros((len(rows), target_width), dtype=torch.long, device=device)
    targets = torch.full((len(rows), target_width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        source = list(row["source_ids"]); target = list(row["target_actions"])
        values = source + [BOS_ID] + target[:-1]
        inputs[index, :len(values)] = torch.tensor(values, device=device)
        positions[index, :len(target)] = torch.arange(len(source), len(source) + len(target), device=device)
        targets[index, :len(target)] = torch.tensor(target, device=device)
    return inputs, positions, targets


@torch.inference_mode()
def _measure(model: Any, rows: list[Mapping[str, Any]], batch_size: int) -> dict[str, Any]:
    by_capability: dict[str, dict[str, int]] = defaultdict(lambda: {"records": 0, "actions": 0, "correct_actions": 0, "exact_sequences": 0})
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs, positions, targets = _packed(batch, torch.device("cuda"))
        logits = model(inputs)
        selected = torch.gather(logits, 1, positions[:, :, None].expand(-1, -1, logits.shape[-1]))
        predictions = selected.argmax(dim=-1)
        valid = targets.ne(-100)
        correct = predictions.eq(targets) & valid
        for index, row in enumerate(batch):
            count = int(valid[index].sum()); hits = int(correct[index].sum()); cap = str(row["capability"])
            by_capability[cap]["records"] += 1; by_capability[cap]["actions"] += count; by_capability[cap]["correct_actions"] += hits; by_capability[cap]["exact_sequences"] += int(hits == count)
    total = {key: sum(value[key] for value in by_capability.values()) for key in ("records", "actions", "correct_actions", "exact_sequences")}
    def rates(value: Mapping[str, int]) -> dict[str, Any]:
        return {**value, "action_accuracy": value["correct_actions"] / value["actions"], "exact_sequence_rate": value["exact_sequences"] / value["records"]}
    return {**rates(total), "per_capability": {key: rates(value) for key, value in sorted(by_capability.items())}}


def _development(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    probes = development_probes(root / protocol["development_catalog"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    rows = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt = f"Capability route: {capability}\n{_semantic_segments(str(probe['prompt']))[-1]}"
        source, _ = tokenizer.encode_source(prompt)
        target = tokenizer.encode_fixed_target(str(teacher[str(probe["probe_id"])]["output"]))
        rows.append({"record_id": str(probe["probe_id"]), "capability": capability, "source_ids": source, "target_actions": target})
    return rows


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("causal-field fit output exists or CUDA unavailable")
    candidate_protocol = _json(root / protocol["candidate_protocol"])
    model, tokenizer = _load_candidate(root, candidate_protocol, root / protocol["candidate_dir"])
    acquisition = _examples(root, candidate_protocol, tokenizer)
    development = _development(root, protocol, tokenizer)
    acquisition_result = _measure(model, acquisition, int(protocol["runtime"]["batch_size"]))
    development_result = _measure(model, development, int(protocol["runtime"]["batch_size"]))
    gap = acquisition_result["action_accuracy"] - development_result["action_accuracy"]
    result = {
        "format": "abi-capability-compiler-phase3-causal-field-fit-result/1",
        "status": "ATTRIBUTED_CONDITIONAL_GENERALIZATION_LIMITED" if acquisition_result["action_accuracy"] >= protocol["reference_gates"]["acquisition_action_accuracy_minimum"] and gap >= protocol["reference_gates"]["action_generalization_gap_minimum"] else "ATTRIBUTED_MIXED_OR_FIT_LIMITED",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": sha256_file(root / protocol["candidate_dir"] / "model.safetensors"),
        "acquisition": acquisition_result,
        "development_teacher_forced": development_result,
        "acquisition_minus_development_action_gap": gap,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "checkpoint_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "claim_boundary": "Read-only fit and conditional-generalization attribution only; no promotion or autonomous-quality effect."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CAUSAL_FIELD_FIT_PROTOCOL_V187.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3/causal_field_fit_v187/result_v188.json"); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
