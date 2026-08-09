"""Audit lexeme-boundary BPE for exact fallback and stable copy pieces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from tokenizers import Tokenizer, models, trainers

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_combined_bpe_feasibility import _architecture_grid, _evaluate
from .capability_compiler_phase3_unicode_span_copy_feasibility import _acquisition_rows


FORMAT = "abi-capability-compiler-phase3-boundary-bpe-feasibility/1"


class BoundaryBpeTokenizer:
    def __init__(self, raw: Any, boundary_split: Any) -> None:
        self.raw = raw
        self.boundary_split = boundary_split
        self.vocab_size = raw.vocab_size
        self.lexeme_to_id = raw.lexeme_to_id
        self.id_to_lexeme = raw.id_to_lexeme

    def split(self, value: bytes | str) -> list[bytes]:
        units = self.boundary_split(value)
        return [piece for unit in units for piece in self.raw.split(unit)]

    def decode_actions(self, actions, source_lexemes):
        return self.raw.decode_actions(actions, source_lexemes)

    def hash(self) -> str:
        return hashlib.sha256((self.raw.hash() + "|UNICODE_LEXEME_BOUNDARIES_V1").encode("ascii")).hexdigest()


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer

    return PortableTokenPlan, Utf8ConcatenativeBpeTokenizer, UnicodeAtomicLexemePointerTokenizer.split


def fit(strings: list[str], vocabulary: int, boundary_split: Any) -> Tokenizer:
    alphabet = sorted({character for value in strings for character in value} | {chr(value) for value in range(0x20, 0x7F)})
    units = [unit.decode("utf-8") for value in strings for unit in boundary_split(value)]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(vocab_size=vocabulary, min_frequency=2, special_tokens=["[UNK]"], initial_alphabet=alphabet, show_progress=False)
    tokenizer.train_from_iterator(units, trainer)
    return tokenizer


def execute(root: Path, protocol_path: Path) -> tuple[dict[str, Any], bytes | None]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("development_used_for_vocabulary") is not False
        or protocol.get("layercake_host_change_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("boundary BPE feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"boundary BPE binding changed: {relative}")

    model_type, raw_type, boundary_split = _types(root, protocol)
    acquisition = _acquisition_rows(root, protocol)
    if len(acquisition) != 14000 or len({row["record_id"] for row in acquisition}) != 14000:
        raise Phase3Error("boundary BPE acquisition identity changed")
    strings = [str(row[field]) for row in acquisition for field in ("prompt", "output")]
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
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
    passing: list[int] = []
    for budget in protocol["bpe"]["budgets"]:
        fitted = fit(strings, int(budget), boundary_split)
        payload = fitted.to_str().encode("utf-8")
        tokenizer = BoundaryBpeTokenizer(raw_type(json.loads(payload)), boundary_split)
        acquisition_report = _evaluate(tokenizer, acquisition)
        heldout_report = _evaluate(tokenizer, heldout)
        maximum_source = max(acquisition_report["maximum_source_actions"], heldout_report["maximum_source_actions"])
        maximum_target = max(acquisition_report["maximum_target_actions"], heldout_report["maximum_target_actions"])
        source_ceiling = ((maximum_source + 31) // 32) * 32
        grid = _architecture_grid(model_type, tokenizer.vocab_size, source_ceiling)
        architecture = min(grid, key=lambda row: (abs(row["parameters"] - target_parameters), row["parameters"]))
        ratio = architecture["parameters"] / target_parameters
        gates = {
            "acquisition_exact": acquisition_report["roundtrip_failures"] == 0,
            "heldout_exact": heldout_report["roundtrip_failures"] == 0,
            "source_bound": maximum_source <= int(protocol["pass_gates"]["maximum_source_actions"]),
            "target_bound": maximum_target <= int(protocol["pass_gates"]["maximum_target_actions"]),
            "matched_capacity": float(protocol["capacity_match"]["minimum_ratio"]) <= ratio <= float(protocol["capacity_match"]["maximum_ratio"]),
            "instruction_pointer_signal": heldout_report["per_capability"]["instruction_following"]["records_with_pointers"] >= int(protocol["pass_gates"]["instruction_records_with_pointers_minimum"]),
        }
        candidates[str(budget)] = {
            "requested_vocabulary": int(budget),
            "actual_fixed_actions": tokenizer.vocab_size,
            "tokenizer_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "boundary_tokenizer_sha256": tokenizer.hash(),
            "acquisition": acquisition_report,
            "heldout": heldout_report,
            "architecture_grid": grid,
            "selected_architecture": architecture,
            "selected_to_native_v94_parameter_ratio": ratio,
            "gates": gates,
            "qualifying": all(gates.values()),
        }
        payloads[int(budget)] = payload
        if all(gates.values()):
            passing.append(int(budget))
    selected = min(passing) if passing else None
    result = {
        "format": "abi-capability-compiler-phase3-boundary-bpe-feasibility-result/1",
        "status": "PASS_BOUNDARY_BPE_FEASIBILITY" if selected is not None else "FAIL_BOUNDARY_BPE_FEASIBILITY",
        "protocol": {"path": protocol_path.name, "sha256": sha256_file(protocol_path)},
        "candidates": candidates,
        "selected_budget": selected,
        "selected_tokenizer_payload_sha256": hashlib.sha256(payloads[selected]).hexdigest() if selected is not None else None,
        "teacher_model_loaded": False,
        "representation_fit_performed": True,
        "neural_training_performed": False,
        "development_used_for_vocabulary": False,
        "layercake_host_changed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-model representation feasibility only. A pass requests a separate LayerCake host construct; it is not neural quality or Phase 3 certification.",
    }
    return result, payloads[selected] if selected is not None else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    directory = root / args.directory
    if directory.exists():
        raise Phase3Error("boundary BPE feasibility directory exists")
    result, payload = execute(root, root / args.protocol)
    directory.mkdir(parents=True)
    _write_immutable(directory / "feasibility.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    if payload is not None:
        _write_immutable(directory / "tokenizer.json", payload)
    print(json.dumps({"status": result["status"], "selected_budget": result["selected_budget"]}, indent=2))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
