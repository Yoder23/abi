"""Fit and audit a combined-acquisition UTF-8 BPE span-copy representation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from tokenizers import Tokenizer, models, trainers

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_bpe_pointer_resilience import _pointer_targets
from .capability_compiler_phase3_unicode_span_copy_feasibility import _acquisition_rows


FORMAT = "abi-capability-compiler-phase3-combined-bpe-feasibility/1"


def _layercake_types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer

    return PortableTokenPlan, Utf8ConcatenativeBpeTokenizer


def fit(strings: list[str], vocabulary: int) -> Tokenizer:
    alphabet = sorted({character for value in strings for character in value} | {chr(value) for value in range(0x20, 0x7F)})
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary,
        min_frequency=2,
        special_tokens=["[UNK]"],
        initial_alphabet=alphabet,
        show_progress=False,
    )
    tokenizer.train_from_iterator(strings, trainer)
    return tokenizer


def _architecture_grid(model_type: Any, vocabulary: int, maximum_source: int) -> list[dict[str, Any]]:
    candidates = [
        (224, 8, 4, 4, 896, 128),
        (256, 8, 3, 3, 1024, 160),
        (256, 8, 4, 4, 1024, 160),
        (288, 8, 3, 3, 1152, 160),
        (288, 8, 4, 4, 1152, 192),
        (320, 8, 3, 3, 1280, 192),
        (320, 8, 4, 4, 1280, 192),
        (352, 8, 3, 3, 1408, 192),
        (352, 8, 4, 4, 1408, 224),
        (384, 8, 3, 3, 1536, 192),
        (384, 8, 4, 4, 1536, 224),
        (416, 8, 3, 3, 1664, 224),
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


def _evaluate(tokenizer: Any, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    maximum_source = maximum_target = failures = 0
    per = {capability: Counter() for capability in CAPABILITIES}
    failure_ids: list[str] = []
    for row in rows:
        capability = str(row["capability"])
        counters = per[capability]
        counters["records"] += 1
        try:
            source = tokenizer.split(str(row["prompt"]))
            output = tokenizer.split(str(row["output"]))
            target = _pointer_targets(source, output, tokenizer.vocab_size, tokenizer)
            exact = tokenizer.decode_actions(target, source) == str(row["output"]).encode("utf-8")
        except (KeyError, ValueError):
            failures += 1
            failure_ids.append(str(row["record_id"]))
            continue
        failures += not exact
        if not exact:
            failure_ids.append(str(row["record_id"]))
        pointers = sum(action >= tokenizer.vocab_size for action in target)
        counters["pointer_actions"] += pointers
        counters["records_with_pointers"] += pointers > 0
        maximum_source = max(maximum_source, len(source))
        maximum_target = max(maximum_target, len(target))
    return {
        "records": len(rows),
        "roundtrip_failures": int(failures),
        "failure_ids": failure_ids[:20],
        "maximum_source_actions": maximum_source,
        "maximum_target_actions": maximum_target,
        "per_capability": {key: dict(value) for key, value in per.items()},
    }


def execute(root: Path, protocol_path: Path) -> tuple[dict[str, Any], bytes | None]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("development_used_for_vocabulary") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("combined BPE feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"combined BPE binding changed: {relative}")

    model_type, tokenizer_type = _layercake_types(root, protocol)
    acquisition = _acquisition_rows(root, protocol)
    if len(acquisition) != 14000 or len({row["record_id"] for row in acquisition}) != 14000:
        raise Phase3Error("combined BPE acquisition identity changed")
    training_strings = [str(row[field]) for row in acquisition for field in ("prompt", "output")]
    teacher = {
        str(row["probe_id"]): row
        for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))
    }
    heldout = [
        {
            "record_id": str(probe["probe_id"]),
            "capability": str(probe["canonical_capability"]),
            "prompt": str(probe["prompt"]),
            "output": str(teacher[str(probe["probe_id"])]["output"]),
        }
        for probe in development_probes(root / protocol["development_catalog"])
    ]

    target_parameters = int(protocol["capacity_match"]["target_parameters"])
    candidates: dict[str, Any] = {}
    payloads: dict[int, bytes] = {}
    qualifying: list[int] = []
    for budget in protocol["bpe"]["budgets"]:
        fitted = fit(training_strings, int(budget))
        payload = fitted.to_str().encode("utf-8")
        tokenizer = tokenizer_type(json.loads(payload))
        acquisition_report = _evaluate(tokenizer, acquisition)
        heldout_report = _evaluate(tokenizer, heldout)
        source_ceiling = int(math.ceil(max(acquisition_report["maximum_source_actions"], heldout_report["maximum_source_actions"]) / 32) * 32)
        grid = _architecture_grid(model_type, tokenizer.vocab_size, source_ceiling)
        selected_architecture = min(grid, key=lambda row: (abs(row["parameters"] - target_parameters), row["parameters"]))
        ratio = selected_architecture["parameters"] / target_parameters
        gates = {
            "acquisition_exact": acquisition_report["roundtrip_failures"] == 0,
            "heldout_exact": heldout_report["roundtrip_failures"] == 0,
            "source_bound": max(acquisition_report["maximum_source_actions"], heldout_report["maximum_source_actions"]) <= int(protocol["pass_gates"]["maximum_source_actions"]),
            "target_bound": max(acquisition_report["maximum_target_actions"], heldout_report["maximum_target_actions"]) <= int(protocol["pass_gates"]["maximum_target_actions"]),
            "matched_capacity": float(protocol["capacity_match"]["minimum_ratio"]) <= ratio <= float(protocol["capacity_match"]["maximum_ratio"]),
            "instruction_pointer_signal": heldout_report["per_capability"]["instruction_following"]["records_with_pointers"] >= int(protocol["pass_gates"]["instruction_records_with_pointers_minimum"]),
        }
        candidates[str(budget)] = {
            "requested_vocabulary": int(budget),
            "actual_fixed_actions": tokenizer.vocab_size,
            "tokenizer_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "host_tokenizer_sha256": tokenizer.hash(),
            "training_alphabet_codepoints": len({character for value in training_strings for character in value} | {chr(value) for value in range(0x20, 0x7F)}),
            "acquisition": acquisition_report,
            "heldout": heldout_report,
            "architecture_grid": grid,
            "selected_architecture": selected_architecture,
            "selected_to_native_v94_parameter_ratio": ratio,
            "gates": gates,
            "qualifying": all(gates.values()),
        }
        payloads[int(budget)] = payload
        if all(gates.values()):
            qualifying.append(int(budget))

    selected_budget = min(qualifying) if qualifying else None
    result = {
        "format": "abi-capability-compiler-phase3-combined-bpe-feasibility-result/1",
        "status": "PASS_COMBINED_BPE_FEASIBILITY" if selected_budget is not None else "FAIL_COMBINED_BPE_FEASIBILITY",
        "protocol": {"path": protocol_path.name, "sha256": sha256_file(protocol_path)},
        "candidates": candidates,
        "selected_budget": selected_budget,
        "selected_tokenizer_payload_sha256": hashlib.sha256(payloads[selected_budget]).hexdigest() if selected_budget is not None else None,
        "teacher_model_loaded": False,
        "representation_fit_performed": True,
        "neural_training_performed": False,
        "development_used_for_vocabulary": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Training-only tokenizer representation, exact-action feasibility, and parameter accounting only; no candidate quality, runtime, or Phase 3 pass is claimed.",
    }
    return result, payloads[selected_budget] if selected_budget is not None else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    directory = root / args.directory
    if directory.exists():
        raise Phase3Error("combined BPE feasibility directory exists")
    result, payload = execute(root, root / args.protocol)
    directory.mkdir(parents=True)
    _write_immutable(directory / "feasibility.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    if payload is not None:
        _write_immutable(directory / "tokenizer.json", payload)
    print(json.dumps({"status": result["status"], "selected_budget": result["selected_budget"]}, indent=2))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
