"""No-training Unicode-atomic lexeme-pointer feasibility and capacity audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_pointer_core import _copy_lexemes


FORMAT = "abi-capability-compiler-phase3-unicode-span-copy-feasibility/1"


def _layercake_types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer

    return PortableTokenPlan, UnicodeAtomicLexemePointerTokenizer


def _targeted_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]


def _acquisition_rows(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_phase1_ir(root / protocol["phase1_ir"]):
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "prompt": str(row["normalized_acquisition_prompt"]),
                "output": str(row["normalized_output"]),
            }
        )
    for row in _targeted_rows(root / protocol["targeted_ir"]):
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "prompt": str(row["host_conformant_acquisition_prompt"]),
                "output": str(row["normalized_output"]),
            }
        )
    return rows


def _build_tokenizer(rows: list[Mapping[str, Any]], tokenizer_type: Any):
    lexemes: set[bytes] = set()
    for row in rows:
        lexemes.update(tokenizer_type.split(str(row["prompt"])))
        lexemes.update(tokenizer_type.split(str(row["output"])))
    return tokenizer_type(sorted(lexemes))


def _encode(tokenizer: Any, prompt: str, output: str) -> dict[str, Any]:
    source_lexemes = tokenizer.split(prompt)
    output_lexemes = tokenizer.split(output)
    copies = _copy_lexemes(source_lexemes, output_lexemes)
    target = tokenizer.encode_target(
        output,
        copy_lexemes=[value.decode("ascii") for value in copies],
        source_lexemes=source_lexemes,
    )
    reconstructed = tokenizer.decode_actions(target, source_lexemes)
    return {
        "source_actions": len(source_lexemes),
        "target_actions": len(target),
        "pointer_actions": sum(action >= tokenizer.vocab_size for action in target),
        "roundtrip": reconstructed == output.encode("utf-8"),
    }


def _architecture_grid(model_type: Any, vocabulary: int, maximum_source: int) -> list[dict[str, Any]]:
    candidates = [
        (160, 8, 4, 4, 640, 96),
        (192, 8, 4, 4, 768, 128),
        (224, 8, 4, 4, 896, 128),
        (256, 8, 3, 3, 1024, 160),
        (256, 8, 4, 4, 1024, 160),
        (288, 8, 3, 3, 1152, 160),
        (288, 8, 4, 4, 1152, 192),
        (320, 8, 3, 3, 1280, 192),
        (320, 8, 4, 4, 1280, 192),
        (352, 8, 3, 3, 1408, 192),
        (384, 8, 3, 3, 1536, 192),
        (384, 8, 4, 4, 1536, 224),
    ]
    values = []
    for width, heads, encoder, decoder, feedforward, pointer in candidates:
        config = {
            "fixed_vocab_size": vocabulary,
            "model_width": width,
            "attention_heads": heads,
            "encoder_layers": encoder,
            "decoder_layers": decoder,
            "feedforward_width": feedforward,
            "pointer_width": pointer,
            "dropout": 0.1,
            "maximum_source_lexemes": maximum_source,
            "maximum_target_actions": 320,
        }
        model = model_type(**config)
        values.append(
            {
                "name": f"w{width}-e{encoder}-d{decoder}",
                **{key: value for key, value in config.items() if key != "fixed_vocab_size"},
                "parameters": model.parameter_count(),
            }
        )
    return values


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("Unicode span-copy feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"Unicode span-copy binding changed: {relative}")

    model_type, tokenizer_type = _layercake_types(root, protocol)
    acquisition = _acquisition_rows(root, protocol)
    if len(acquisition) != 14000 or len({row["record_id"] for row in acquisition}) != 14000:
        raise Phase3Error("combined acquisition identity changed")
    tokenizer = _build_tokenizer(acquisition, tokenizer_type)

    reports: dict[str, Any] = {}
    maximum_source = maximum_target = 0
    groups: dict[str, list[dict[str, Any]]] = {"acquisition": acquisition, "heldout": []}
    teacher = {
        str(row["probe_id"]): row
        for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))
    }
    for probe in development_probes(root / protocol["development_catalog"]):
        groups["heldout"].append(
            {
                "record_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "prompt": str(probe["prompt"]),
                "output": str(teacher[str(probe["probe_id"])]["output"]),
            }
        )

    for group, rows in groups.items():
        per = {capability: Counter() for capability in CAPABILITIES}
        failures = 0
        for row in rows:
            encoded = _encode(tokenizer, str(row["prompt"]), str(row["output"]))
            maximum_source = max(maximum_source, encoded["source_actions"])
            maximum_target = max(maximum_target, encoded["target_actions"])
            failures += not encoded["roundtrip"]
            counters = per[str(row["capability"])]
            counters["records"] += 1
            counters["pointer_actions"] += encoded["pointer_actions"]
            counters["records_with_pointers"] += encoded["pointer_actions"] > 0
        reports[group] = {
            "records": len(rows),
            "roundtrip_failures": int(failures),
            "per_capability": {key: dict(value) for key, value in per.items()},
        }

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
        "status": "PASS_UNICODE_SPAN_COPY_FEASIBILITY" if all(gates.values()) else "FAIL_UNICODE_SPAN_COPY_FEASIBILITY",
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
        "claim_boundary": "Exact Unicode representation and parameter accounting only; no candidate quality, runtime, or Phase 3 pass is claimed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("Unicode span-copy feasibility output exists")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
