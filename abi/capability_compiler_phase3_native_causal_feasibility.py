"""No-model feasibility for native-token causal target-state transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, load_phase1_ir


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY" or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("tensor_extraction_authorized") is not False:
        raise Phase3Error("native causal feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"native causal feasibility binding changed: {relative}")
    tokenizer = Tokenizer.from_file(protocol["source"]["tokenizer_json"])
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    actions = 0
    maximum = 0
    terminal_matches = 0
    exact = 0
    order = hashlib.sha256()
    by_capability = Counter()
    for row in rows:
        ids = tokenizer.encode(str(row["normalized_output"]), add_special_tokens=False).ids
        normalized = [int(value) for value in row["normalized_output_token_ids"]]
        authoritative = [int(value) for value in row["authoritative_generated_token_ids"]]
        if tokenizer.decode(ids, skip_special_tokens=False) != str(row["normalized_output"]):
            raise Phase3Error("native response roundtrip changed")
        exact += int(ids == normalized)
        terminal_matches += int(authoritative == normalized + [int(protocol["source"]["terminal_token_id"])])
        actions += len(ids)
        maximum = max(maximum, len(ids))
        by_capability[str(row["capability"])] += len(ids)
        order.update((str(row["ir_record_id"]) + ":" + ",".join(map(str, ids)) + "\n").encode())
    payload = actions * int(protocol["projection"]["target_width"]) * 2
    passed = exact == len(rows) and terminal_matches == len(rows) and payload <= int(protocol["selection"]["payload_bytes_maximum"]) and set(by_capability) == set(protocol["capabilities"])
    return {"format": "abi-capability-compiler-phase3-native-causal-feasibility/1", "status": "PASS_FEASIBLE" if passed else "FAIL_FEASIBILITY", "records": len(rows), "exact_native_action_sequences": exact, "exact_authoritative_terminal_sequences": terminal_matches, "target_actions": actions, "maximum_target_actions_excluding_host_eos": maximum, "causal_predecessor_states_available": actions, "projected_width": protocol["projection"]["target_width"], "projected_fp16_payload_bytes": payload, "offset_payload_bytes": (len(rows) + 1) * 8, "record_action_order_sha256": order.hexdigest(), "actions_by_capability": dict(sorted(by_capability.items())), "teacher_model_loaded": False, "tensor_values_extracted": False, "neural_training_performed": False, "phase3_certified": False, "final_test_accessed": False, "next_gate": "Preregister one exact GPU extraction of causal predecessor states for every native response action." if passed else "Close native causal state branch."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_FEASIBILITY_PROTOCOL_V88.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_native_causal/feasibility_v88.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("native causal feasibility output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
