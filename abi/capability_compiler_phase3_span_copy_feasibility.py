"""No-model exact UTF-8 BPE/lexeme-copy feasibility and capacity audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zipfile

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json, _layercake_api, _tokenizer
from .capability_compiler_phase3_bpe_pointer_resilience import _pointer_targets
from .capability_compiler_phase3_route_bridge import _select_controls
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-span-copy-feasibility/1"


def _targeted_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]


def _encode(tokenizer: Any, control: bytes, prompt: str, output: str) -> dict[str, Any]:
    source_lexemes = [control] + tokenizer.split("\n" + prompt)
    source_ids = [tokenizer.lexeme_to_id[piece] for piece in source_lexemes]
    output_lexemes = tokenizer.split(output)
    target = _pointer_targets(source_lexemes, output_lexemes, tokenizer.vocab_size, tokenizer)
    reconstructed = tokenizer.decode_actions(target, source_lexemes)
    return {
        "source_actions": len(source_ids),
        "target_actions": len(target),
        "pointer_actions": sum(action >= tokenizer.vocab_size for action in target),
        "roundtrip": reconstructed == output.encode("utf-8"),
    }


def _architecture_grid(model_type: Any, vocabulary: int, maximum_source: int) -> list[dict[str, Any]]:
    candidates = [
        {"name": "w256-e4-d4", "model_width": 256, "attention_heads": 8, "encoder_layers": 4, "decoder_layers": 4, "feedforward_width": 1024, "pointer_width": 160},
        {"name": "w320-e4-d3", "model_width": 320, "attention_heads": 8, "encoder_layers": 4, "decoder_layers": 3, "feedforward_width": 1280, "pointer_width": 192},
        {"name": "w320-e3-d4", "model_width": 320, "attention_heads": 8, "encoder_layers": 3, "decoder_layers": 4, "feedforward_width": 1280, "pointer_width": 192},
        {"name": "w384-e3-d3", "model_width": 384, "attention_heads": 8, "encoder_layers": 3, "decoder_layers": 3, "feedforward_width": 1536, "pointer_width": 192},
    ]
    common = {"fixed_vocab_size": vocabulary, "dropout": 0.1, "maximum_source_lexemes": maximum_source, "maximum_target_actions": 320}
    values = []
    for candidate in candidates:
        model = model_type(**common, **{key: value for key, value in candidate.items() if key != "name"})
        values.append({**candidate, "maximum_source_lexemes": maximum_source, "maximum_target_actions": 320, "parameters": model.parameter_count()})
    return values


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("span-copy feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"span-copy binding changed: {relative}")
    base = _json(root / protocol["bpe_reference_protocol"])
    _, model_type, tokenizer_type, _, _ = _layercake_api(root, base)
    tokenizer = _tokenizer(root, base, tokenizer_type)
    original = load_phase1_ir(root / base["phase1_ir"])
    controls = _select_controls(original, tokenizer)
    control_by_capability = {capability: controls[index][1] for index, capability in enumerate(CAPABILITIES)}
    targeted = _targeted_rows(root / protocol["targeted_ir"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / base["teacher_reference"], encoding="utf-8"))}
    heldout_probes = development_probes(root / base["development_catalog"])
    groups: dict[str, list[tuple[str, str, str]]] = {"acquisition": [], "heldout": []}
    for row in original:
        body = "\n".join(str(row["normalized_acquisition_prompt"]).splitlines()[1:]).strip()
        groups["acquisition"].append((str(row["capability"]), body, str(row["normalized_output"])))
    for row in targeted:
        groups["acquisition"].append((str(row["capability"]), str(row["host_conformant_acquisition_prompt"]), str(row["normalized_output"])))
    for probe in heldout_probes:
        capability = str(probe["canonical_capability"])
        groups["heldout"].append((capability, _semantic_segments(str(probe["prompt"]))[-1], str(teacher[str(probe["probe_id"])] ["output"])))
    reports: dict[str, Any] = {}
    maximum_source = maximum_target = 0
    for group, records in groups.items():
        per = {capability: Counter() for capability in CAPABILITIES}
        failures = 0
        for capability, prompt, output in records:
            encoded = _encode(tokenizer, control_by_capability[capability], prompt, output)
            maximum_source = max(maximum_source, encoded["source_actions"])
            maximum_target = max(maximum_target, encoded["target_actions"])
            failures += not encoded["roundtrip"]
            per[capability]["records"] += 1
            per[capability]["pointer_actions"] += encoded["pointer_actions"]
            per[capability]["records_with_pointers"] += encoded["pointer_actions"] > 0
        reports[group] = {"records": len(records), "roundtrip_failures": int(failures), "per_capability": {key: dict(value) for key, value in per.items()}}
    source_ceiling = int(math.ceil(maximum_source / 32) * 32)
    grid = _architecture_grid(model_type, tokenizer.vocab_size, source_ceiling)
    target_parameters = int(protocol["capacity_match"]["target_parameters"])
    selected = min(grid, key=lambda row: (abs(row["parameters"] - target_parameters), row["parameters"]))
    ratio = selected["parameters"] / target_parameters
    gates = {
        "acquisition_exact": reports["acquisition"]["roundtrip_failures"] == 0,
        "heldout_exact": reports["heldout"]["roundtrip_failures"] == 0,
        "source_bound": maximum_source <= int(protocol["pass_gates"]["maximum_source_actions"]),
        "target_bound": maximum_target <= int(protocol["pass_gates"]["maximum_target_actions"]),
        "matched_capacity": float(protocol["capacity_match"]["minimum_ratio"]) <= ratio <= float(protocol["capacity_match"]["maximum_ratio"]),
        "instruction_pointer_signal": reports["heldout"]["per_capability"]["instruction_following"]["records_with_pointers"] >= int(protocol["pass_gates"]["instruction_records_with_pointers_minimum"]),
    }
    return {
        "format": FORMAT,
        "status": "PASS_SPAN_COPY_FEASIBILITY" if all(gates.values()) else "FAIL_SPAN_COPY_FEASIBILITY",
        "tokenizer_sha256": tokenizer.hash(),
        "fixed_actions": tokenizer.vocab_size,
        "maxima": {"source_actions": maximum_source, "target_actions": maximum_target, "selected_source_ceiling": source_ceiling},
        "reports": reports,
        "architecture_grid": grid,
        "selected_architecture": selected,
        "selected_to_native_v94_parameter_ratio": ratio,
        "gates": gates,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Exact representation and parameter accounting only; no candidate quality, runtime, or Phase 3 pass is claimed."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists():
        raise Phase3Error("span-copy feasibility output exists")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
