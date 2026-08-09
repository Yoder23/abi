"""Audit exact native-token source pointers without loading or training a model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_combined_coverage import broad_examples
from .capability_compiler_phase3_native_causal_core import load_protocol
from .capability_compiler_phase3_teacher_native_core import _examples, _layercake_api, _tokenizer, controlled_prompt
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-native-pointer-feasibility/1"


def pointer_encode(source: list[int], target: list[int], fixed_vocab_size: int) -> tuple[list[int], int]:
    positions: dict[int, list[int]] = defaultdict(list)
    for index, action in enumerate(source):
        positions[int(action)].append(index)
    encoded: list[int] = []
    pointers = 0
    for action in target:
        matches = positions.get(int(action), [])
        if action != 2 and len(matches) == 1:
            encoded.append(fixed_vocab_size + matches[0])
            pointers += 1
        else:
            encoded.append(int(action))
    return encoded, pointers


def pointer_decode(actions: Iterable[int], source: list[int], fixed_vocab_size: int) -> list[int]:
    decoded = []
    for action in actions:
        value = int(action)
        if value >= fixed_vocab_size:
            position = value - fixed_vocab_size
            if not 0 <= position < len(source):
                raise Phase3Error("pointer outside source")
            decoded.append(int(source[position]))
        else:
            decoded.append(value)
    return decoded


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("native pointer feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"native pointer binding changed: {relative}")
    candidate, _ = load_protocol(root, root / protocol["candidate_protocol"])
    _, tokenizer_type, _, _ = _layercake_api(root, candidate)
    tokenizer = _tokenizer(root, candidate, tokenizer_type)
    acquisition = _examples(root, candidate, tokenizer)
    ir_path = root / protocol["targeted_ir"]
    with zipfile.ZipFile(ir_path) as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]
    acquisition.extend(broad_examples(rows, tokenizer, maximum_source_lexemes=192, maximum_target_actions=320))
    probes = development_probes(root / candidate["development_catalog"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / candidate["teacher_reference"], encoding="utf-8"))}
    heldout = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        source, _ = tokenizer.encode_source(controlled_prompt(capability, _semantic_segments(str(probe["prompt"]))[-1]))
        target = tokenizer.encode_fixed_target(str(teacher[str(probe["probe_id"])] ["output"]))
        heldout.append({"record_id": str(probe["probe_id"]), "capability": capability, "source_ids": source, "target_actions": target})
    known = {capability: set() for capability in CAPABILITIES}
    for row in acquisition:
        known[str(row["capability"])].update(int(value) for value in row["target_actions"])
    reports = {}
    for name, records in (("acquisition", acquisition), ("heldout", heldout)):
        per = {capability: Counter() for capability in CAPABILITIES}
        failures = 0
        for row in records:
            capability = str(row["capability"])
            target = [int(value) for value in row["target_actions"]]
            hybrid, pointers = pointer_encode(row["source_ids"], target, tokenizer.vocab_size)
            if pointer_decode(hybrid, row["source_ids"], tokenizer.vocab_size) != target:
                failures += 1
            unseen = [value for value in target if value not in known[capability]] if name == "heldout" else []
            pointer_positions = {value for value, positions in Counter(row["source_ids"]).items() if positions == 1}
            per[capability]["records"] += 1
            per[capability]["target_actions"] += len(target)
            per[capability]["pointer_actions"] += pointers
            per[capability]["unseen_actions"] += len(unseen)
            per[capability]["unseen_pointerable_actions"] += sum(value in pointer_positions for value in unseen)
        reports[name] = {
            "records": len(records),
            "roundtrip_failures": failures,
            "per_capability": {capability: dict(per[capability]) for capability in CAPABILITIES},
        }
    instruction = reports["heldout"]["per_capability"]["instruction_following"]
    ratio = instruction["unseen_pointerable_actions"] / max(1, instruction["unseen_actions"])
    reports["heldout"]["instruction_unseen_pointerable_ratio"] = ratio
    passed = reports["acquisition"]["roundtrip_failures"] == 0 and reports["heldout"]["roundtrip_failures"] == 0 and ratio >= float(protocol["pass_gates"]["instruction_unseen_pointerable_ratio_minimum"])
    return {
        "format": FORMAT,
        "status": "PASS_NATIVE_POINTER_FEASIBILITY" if passed else "FAIL_NATIVE_POINTER_FEASIBILITY",
        "fixed_vocab_size": tokenizer.vocab_size,
        "model_parameter_change_required": 0,
        "reports": reports,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Representation feasibility and failure attribution only; no host interface, trained model, or quality pass is claimed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(); output = root / args.output
    if output.exists():
        raise Phase3Error("native pointer feasibility output exists")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
