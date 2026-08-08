"""Read-only fit and autonomous-prefix attribution for failed V94."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_native_causal_core import _load_candidate, load_protocol
from .capability_compiler_phase3_teacher_native_core import _examples, _json, _layercake_api
from .capability_compiler_phase3_teacher_native_fit import _development_examples, _measure


def _common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("status") != "PREREGISTERED_READ_ONLY_FIT_AND_PREFIX_ATTRIBUTION" or protocol.get("neural_training_authorized") is not False or protocol.get("teacher_model_loading_authorized") is not False:
        raise Phase3Error("native causal fit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"native causal fit binding changed: {relative}")
    candidate_protocol, _ = load_protocol(root, (root / protocol["candidate_protocol"]).resolve())
    candidate = (root / protocol["candidate_dir"]).resolve()
    model = _load_candidate(root, candidate_protocol, candidate)
    _, tokenizer_type, _, _ = _layercake_api(root, candidate_protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    training = _measure(model, tokenizer, _examples(root, candidate_protocol, tokenizer), int(protocol["batch_size"]))
    development = _measure(model, tokenizer, _development_examples(root, candidate_protocol, tokenizer), int(protocol["batch_size"]))
    autonomous = [json.loads(line) for line in (root / protocol["autonomous_outputs"]).read_text(encoding="utf-8").splitlines()]
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / candidate_protocol["teacher_reference"], encoding="utf-8"))}
    totals = Counter()
    collapsed_prefix = []
    noncollapsed_prefix = []
    for row in autonomous:
        predicted = tokenizer.encode_fixed_target(str(row["output"]))
        expected = tokenizer.encode_fixed_target(str(teacher[str(row["probe_id"])]["output"]))
        prefix = _common_prefix(predicted, expected)
        denominator = max(1, len(expected) - 1)
        ratio = min(prefix, denominator) / denominator
        totals["sequences"] += 1
        totals["exact"] += int(predicted == expected)
        totals["zero_prefix"] += int(prefix == 0)
        totals["prefix_micros"] += round(ratio * 1_000_000)
        (collapsed_prefix if row["repetition_collapse"] else noncollapsed_prefix).append(ratio)
    train_acc = training["overall"]["action_accuracy"]
    dev_acc = development["overall"]["action_accuracy"]
    return {
        "format": "abi-capability-compiler-phase3-native-causal-fit/1",
        "status": "PASS_READ_ONLY_ATTRIBUTION",
        "candidate_checkpoint_sha256": sha256_file(candidate / "model.safetensors"),
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "training_fit": training,
        "development_teacher_forced_fit": development,
        "action_generalization_gap": train_acc - dev_acc,
        "autonomous_prefix": {
            "sequences": totals["sequences"],
            "exact_teacher_sequences": totals["exact"],
            "zero_matching_prefix": totals["zero_prefix"],
            "mean_teacher_prefix_fraction": totals["prefix_micros"] / 1_000_000 / totals["sequences"],
            "collapsed_mean_prefix_fraction": sum(collapsed_prefix) / len(collapsed_prefix) if collapsed_prefix else None,
            "noncollapsed_mean_prefix_fraction": sum(noncollapsed_prefix) / len(noncollapsed_prefix) if noncollapsed_prefix else None,
        },
        "attribution_rule": protocol["attribution_rule"],
        "phase3_certified": False,
        "final_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_FIT_PROTOCOL_V96.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_native_causal_core/fit_v96.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("native causal fit output exists")
    result = run(root, (root / args.protocol).resolve()); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
