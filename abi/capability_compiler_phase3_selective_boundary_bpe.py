"""Audit identifier-selective boundary BPE under unchanged Phase 3 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_boundary_bpe_feasibility import BoundaryBpeTokenizer
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_combined_bpe_feasibility import _architecture_grid, _evaluate, fit
from .capability_compiler_phase3_unicode_span_copy_feasibility import _acquisition_rows


FORMAT = "abi-capability-compiler-phase3-selective-boundary-bpe/1"
PROTECTED = re.compile(rb"^[A-Za-z0-9_]+$")


def selective_split(value: bytes | str, unicode_split: Any, raw_split: Any) -> list[bytes]:
    units = unicode_split(value)
    pieces: list[bytes] = []
    buffered = bytearray()
    def flush() -> None:
        if buffered:
            pieces.extend(raw_split(bytes(buffered)))
            buffered.clear()
    for unit in units:
        protected = PROTECTED.fullmatch(unit) is not None and (b"_" in unit or any(48 <= byte <= 57 for byte in unit))
        if protected:
            flush()
            pieces.extend(raw_split(unit))
        else:
            buffered.extend(unit)
    flush()
    return pieces


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer
    return PortableTokenPlan, Utf8ConcatenativeBpeTokenizer, UnicodeAtomicLexemePointerTokenizer.split


def execute(root: Path, protocol_path: Path) -> tuple[dict[str, Any], bytes | None]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False or protocol.get("development_used_for_vocabulary") is not False or protocol.get("layercake_host_change_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("selective boundary BPE governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"selective boundary binding changed: {relative}")
    model_type, raw_type, unicode_split = _types(root, protocol)
    acquisition = _acquisition_rows(root, protocol)
    if len(acquisition) != 14000 or len({row["record_id"] for row in acquisition}) != 14000:
        raise Phase3Error("selective boundary acquisition identity changed")
    strings = [str(row[field]) for row in acquisition for field in ("prompt", "output")]
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    heldout = [{"record_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "prompt": str(probe["prompt"]), "output": str(teacher[str(probe["probe_id"])]["output"])} for probe in development_probes(root / protocol["development_catalog"])]
    target_parameters = int(protocol["capacity_match"]["target_parameters"])
    candidates: dict[str, Any] = {}
    payloads: dict[int, bytes] = {}
    passing: list[int] = []
    for budget in protocol["bpe"]["budgets"]:
        fitted = fit(strings, int(budget))
        payload = fitted.to_str().encode("utf-8")
        raw = raw_type(json.loads(payload))
        splitter = lambda value, u=unicode_split, r=raw.split: selective_split(value, u, r)
        tokenizer = BoundaryBpeTokenizer(raw, splitter)
        acq = _evaluate(tokenizer, acquisition)
        dev = _evaluate(tokenizer, heldout)
        maximum_source = max(acq["maximum_source_actions"], dev["maximum_source_actions"])
        maximum_target = max(acq["maximum_target_actions"], dev["maximum_target_actions"])
        source_ceiling = ((maximum_source + 31) // 32) * 32
        grid = _architecture_grid(model_type, tokenizer.vocab_size, source_ceiling)
        architecture = min(grid, key=lambda row: (abs(row["parameters"] - target_parameters), row["parameters"]))
        ratio = architecture["parameters"] / target_parameters
        gates = {"acquisition_exact": acq["roundtrip_failures"] == 0, "heldout_exact": dev["roundtrip_failures"] == 0, "source_bound": maximum_source <= int(protocol["pass_gates"]["maximum_source_actions"]), "target_bound": maximum_target <= int(protocol["pass_gates"]["maximum_target_actions"]), "matched_capacity": float(protocol["capacity_match"]["minimum_ratio"]) <= ratio <= float(protocol["capacity_match"]["maximum_ratio"]), "instruction_pointer_signal": dev["per_capability"]["instruction_following"]["records_with_pointers"] >= int(protocol["pass_gates"]["instruction_records_with_pointers_minimum"])}
        candidates[str(budget)] = {"requested_vocabulary": int(budget), "actual_fixed_actions": tokenizer.vocab_size, "tokenizer_payload_sha256": hashlib.sha256(payload).hexdigest(), "selective_tokenizer_sha256": tokenizer.hash(), "acquisition": acq, "heldout": dev, "architecture_grid": grid, "selected_architecture": architecture, "selected_to_native_v94_parameter_ratio": ratio, "gates": gates, "qualifying": all(gates.values())}
        payloads[int(budget)] = payload
        if all(gates.values()): passing.append(int(budget))
    selected = min(passing) if passing else None
    result = {"format": "abi-capability-compiler-phase3-selective-boundary-bpe-result/1", "status": "PASS_SELECTIVE_BOUNDARY_BPE" if selected is not None else "FAIL_SELECTIVE_BOUNDARY_BPE", "protocol": {"path": protocol_path.name, "sha256": sha256_file(protocol_path)}, "candidates": candidates, "selected_budget": selected, "selected_tokenizer_payload_sha256": hashlib.sha256(payloads[selected]).hexdigest() if selected is not None else None, "teacher_model_loaded": False, "representation_fit_performed": True, "neural_training_performed": False, "development_used_for_vocabulary": False, "layercake_host_changed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "No-model selective-boundary feasibility only; a pass authorizes only a separate host construct."}
    return result, payloads[selected] if selected is not None else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); parser.add_argument("--directory", required=True); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); directory = root / args.directory
    if directory.exists(): raise Phase3Error("selective boundary output exists")
    result, payload = execute(root, root / args.protocol); directory.mkdir(parents=True)
    _write_immutable(directory / "feasibility.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    if payload is not None: _write_immutable(directory / "tokenizer.json", payload)
    print(json.dumps({"status": result["status"], "selected_budget": result["selected_budget"]}, indent=2)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
