"""Read-only teacher-forced fit diagnostic for sealed V170."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_route_bridge import _collate
from .capability_compiler_phase3_selective_bpe_core import _examples, _load_candidate


FORMAT = "abi-capability-compiler-phase3-selective-bpe-fit-diagnostic/1"


@torch.inference_mode()
def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("neural_training_authorized") is not False or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("fit diagnostic governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"fit diagnostic binding changed: {relative}")
    candidate_protocol = _json(root / protocol["candidate_protocol"])
    candidate = root / protocol["candidate_dir"]
    model, tokenizer = _load_candidate(root, candidate_protocol, candidate)
    examples, _ = _examples(root, candidate_protocol, tokenizer)
    totals = {capability: Counter() for capability in CAPABILITIES}
    for start in range(0, len(examples), int(protocol["batch_size"])):
        batch = examples[start:start + int(protocol["batch_size"])]
        source, targets = _collate(batch, torch.device("cuda")); predicted = model(source, targets)["log_probs"].argmax(-1); mask = targets.ge(0); matches = predicted.eq(targets) & mask
        for index, row in enumerate(batch):
            count = int(mask[index].sum()); correct = int(matches[index].sum()); values = totals[row["capability"]]; values["records"] += 1; values["actions"] += count; values["correct_actions"] += correct; values["exact_sequences"] += correct == count
    actions = sum(value["actions"] for value in totals.values()); correct = sum(value["correct_actions"] for value in totals.values()); records = sum(value["records"] for value in totals.values()); exact = sum(value["exact_sequences"] for value in totals.values()); action_accuracy = correct / actions; exact_rate = exact / records
    gates = {"action_fit_reference": action_accuracy >= float(protocol["attribution_thresholds"]["action_accuracy_minimum"]), "sequence_fit_reference": exact_rate >= float(protocol["attribution_thresholds"]["exact_sequence_rate_minimum"])}
    return {"format": "abi-capability-compiler-phase3-selective-bpe-fit-diagnostic-result/1", "status": "ATTRIBUTED_GENERALIZATION_LIMITED" if all(gates.values()) else "ATTRIBUTED_TRAINING_FIT_LIMITED", "records": records, "actions": actions, "correct_actions": correct, "action_accuracy": action_accuracy, "exact_sequences": exact, "exact_sequence_rate": exact_rate, "per_capability": {key: {**dict(value), "action_accuracy": value["correct_actions"] / value["actions"], "exact_sequence_rate": value["exact_sequences"] / value["records"]} for key, value in totals.items()}, "gates": gates, "checkpoint_changed": False, "neural_training_performed": False, "teacher_model_loaded": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Read-only teacher-forced fit attribution; no autonomous quality or promotion effect."}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("fit diagnostic output exists")
    result = execute(root, root / args.protocol); output.parent.mkdir(parents=True, exist_ok=True); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps({key: result[key] for key in ("status", "action_accuracy", "exact_sequence_rate")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
