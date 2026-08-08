"""Read-only train/development fit attribution for the failed V75 candidate."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_teacher_native_core import _collate, _examples, _json, _layercake_api, _load_candidate, controlled_prompt, load_protocol


def summarize_counts(actions: int, correct: int, sequences: int, exact: int, nll_sum: float, pointer_predictions: int) -> dict[str, Any]:
    return {"actions": actions, "correct_actions": correct, "action_accuracy": correct / actions, "sequences": sequences, "exact_sequences": exact, "exact_sequence_rate": exact / sequences, "mean_nll": nll_sum / actions, "perplexity": math.exp(min(50.0, nll_sum / actions)), "pointer_argmax_actions": pointer_predictions}


def _development_examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    examples = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt = controlled_prompt(capability, _semantic_segments(str(probe["prompt"]))[-1])
        source, _ = tokenizer.encode_source(prompt)
        target = tokenizer.encode_fixed_target(str(teacher[str(probe["probe_id"])]["output"]))
        examples.append({"record_id": str(probe["probe_id"]), "capability": capability, "source_ids": source, "target_actions": target})
    return examples


@torch.inference_mode()
def _measure(model, tokenizer, rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, Any]:
    totals = Counter()
    by_capability: dict[str, Counter] = defaultdict(Counter)
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        source, targets = _collate(batch, device)
        result = model(source, targets)["log_probs"].float()
        mask = targets.ge(0)
        safe = targets.clamp(min=0)
        predicted = result.argmax(dim=-1)
        correct = predicted.eq(targets) & mask
        losses = F.nll_loss(result.reshape(-1, result.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="none").reshape_as(targets)
        for index, row in enumerate(batch):
            row_mask = mask[index]
            actions = int(row_mask.sum())
            row_correct = int(correct[index][row_mask].sum())
            exact = int(row_correct == actions)
            nll = float(losses[index][row_mask].sum())
            pointers = int((predicted[index][row_mask] >= tokenizer.vocab_size).sum())
            values = {"actions": actions, "correct": row_correct, "sequences": 1, "exact": exact, "nll_micros": round(nll * 1_000_000), "pointers": pointers}
            totals.update(values)
            by_capability[str(row["capability"])].update(values)
    def finish(value: Counter):
        return summarize_counts(value["actions"], value["correct"], value["sequences"], value["exact"], value["nll_micros"] / 1_000_000, value["pointers"])
    return {"overall": finish(totals), "per_capability": {name: finish(by_capability[name]) for name in CAPABILITIES}}


def run(root: Path, fit_protocol_path: Path) -> dict[str, Any]:
    fit_protocol = _json(fit_protocol_path)
    if fit_protocol.get("status") != "PREREGISTERED_READ_ONLY_FIT_ATTRIBUTION" or fit_protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("fit attribution governance changed")
    for relative, expected in fit_protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"fit attribution binding changed: {relative}")
    candidate_protocol, _ = load_protocol(root, (root / fit_protocol["candidate_protocol"]).resolve())
    candidate = (root / fit_protocol["candidate_dir"]).resolve()
    model, tokenizer = _load_candidate(root, candidate_protocol, candidate)
    train_rows = _examples(root, candidate_protocol, tokenizer)
    development_rows = _development_examples(root, candidate_protocol, tokenizer)
    training = _measure(model, tokenizer, train_rows, int(fit_protocol["batch_size"]))
    development = _measure(model, tokenizer, development_rows, int(fit_protocol["batch_size"]))
    autonomous = [json.loads(line) for line in (root / fit_protocol["autonomous_outputs"]).read_text(encoding="utf-8").splitlines()]
    errors = Counter(row["generation_error"] for row in autonomous if row["generation_error"] is not None)
    return {"format": "abi-capability-compiler-phase3-teacher-native-fit/1", "status": "PASS_READ_ONLY_ATTRIBUTION", "candidate_checkpoint_sha256": sha256_file(candidate / "model.safetensors"), "teacher_model_loaded": False, "neural_training_performed": False, "training_fit": training, "development_teacher_forced_fit": development, "autonomous": {"observations": len(autonomous), "functional_passes": sum(row["functional_pass"] for row in autonomous), "repetition_collapses": sum(row["repetition_collapse"] for row in autonomous), "generation_errors": sum(errors.values()), "error_types": dict(errors)}, "attribution_rule": "Training exactness below 95% indicates residual fit capacity. A training-development action gap above 10 points indicates held-out conditional generalization failure. Development action accuracy above 95% with poor autonomous results indicates exposure-bias/state-recovery failure.", "phase3_certified": False, "final_test_accessed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_NATIVE_FIT_PROTOCOL_V78.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_teacher_native_core/fit_v78.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("fit attribution output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
