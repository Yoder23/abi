"""No-model acquisition-versus-held-out structural coverage audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_teacher_native_core import _examples, _json, _layercake_api, _tokenizer, controlled_prompt, load_protocol as load_candidate_protocol


def _ngrams(values: Sequence[int], width: int) -> Iterable[tuple[int, ...]]:
    return (tuple(values[index:index + width]) for index in range(max(0, len(values) - width + 1)))


def _coverage(rows: Sequence[Mapping[str, Any]], known: Mapping[str, set[tuple[int, ...]]], field: str, width: int) -> dict[str, Any]:
    totals = Counter()
    by_capability: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        capability = str(row["capability"])
        for value in _ngrams(row[field], width):
            totals["total"] += 1; by_capability[capability]["total"] += 1
            if value in known[capability]:
                totals["seen"] += 1; by_capability[capability]["seen"] += 1
    def finish(value: Counter) -> dict[str, Any]:
        return {"total": value["total"], "seen": value["seen"], "unseen": value["total"] - value["seen"], "coverage": value["seen"] / value["total"] if value["total"] else 1.0}
    return {"overall": finish(totals), "per_capability": {name: finish(by_capability[name]) for name in CAPABILITIES}}


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("status") != "PREREGISTERED_NO_MODEL_ACQUISITION_COVERAGE_AUDIT" or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("acquisition coverage governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"acquisition coverage binding changed: {relative}")
    candidate_protocol, _ = load_candidate_protocol(root, (root / protocol["candidate_protocol"]).resolve())
    _, tokenizer_type, _, _ = _layercake_api(root, candidate_protocol)
    tokenizer = _tokenizer(root, candidate_protocol, tokenizer_type)
    acquisition = _examples(root, candidate_protocol, tokenizer)
    probes = development_probes((root / candidate_protocol["development_catalog"]).resolve())
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / candidate_protocol["teacher_reference"], encoding="utf-8"))}
    heldout = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        source, _ = tokenizer.encode_source(controlled_prompt(capability, _semantic_segments(str(probe["prompt"]))[-1]))
        target = tokenizer.encode_fixed_target(str(teacher[str(probe["probe_id"])]["output"]))
        heldout.append({"record_id": str(probe["probe_id"]), "capability": capability, "source_ids": source, "target_actions": target})
    reports = {}
    for field in ("source_ids", "target_actions"):
        reports[field] = {}
        for width in (1, 2, 3, 4):
            known = {name: set() for name in CAPABILITIES}
            for row in acquisition:
                known[str(row["capability"])].update(_ngrams(row[field], width))
            reports[field][str(width)] = _coverage(heldout, known, field, width)
    exact_source = {name: set() for name in CAPABILITIES}; exact_target = {name: set() for name in CAPABILITIES}
    source_lengths = {name: [] for name in CAPABILITIES}; target_lengths = {name: [] for name in CAPABILITIES}
    for row in acquisition:
        cap = str(row["capability"]); exact_source[cap].add(tuple(row["source_ids"])); exact_target[cap].add(tuple(row["target_actions"])); source_lengths[cap].append(len(row["source_ids"])); target_lengths[cap].append(len(row["target_actions"]))
    exact = {"source_sequences": sum(tuple(row["source_ids"]) in exact_source[str(row["capability"])] for row in heldout), "target_sequences": sum(tuple(row["target_actions"]) in exact_target[str(row["capability"])] for row in heldout), "observations": len(heldout)}
    outside = {"source_length": 0, "target_length": 0}
    for row in heldout:
        cap = str(row["capability"])
        outside["source_length"] += int(not min(source_lengths[cap]) <= len(row["source_ids"]) <= max(source_lengths[cap]))
        outside["target_length"] += int(not min(target_lengths[cap]) <= len(row["target_actions"]) <= max(target_lengths[cap]))
    source_trigram = reports["source_ids"]["3"]["overall"]["coverage"]
    target_fourgram = reports["target_actions"]["4"]["overall"]["coverage"]
    threshold = float(protocol["decision_rule"]["minimum_coverage"])
    gap = source_trigram < threshold or target_fourgram < threshold
    return {"format": "abi-capability-compiler-phase3-acquisition-coverage/1", "status": "PASS_COVERAGE_GAP_MEASURED" if gap else "PASS_NO_MATERIAL_NGRAM_COVERAGE_GAP", "records": {"acquisition": len(acquisition), "heldout": len(heldout)}, "exact_sequence_overlap": exact, "outside_acquisition_length_range": outside, "ngram_coverage": reports, "headline": {"source_trigram_coverage": source_trigram, "target_fourgram_coverage": target_fourgram, "minimum_coverage": threshold, "material_gap": gap}, "teacher_model_loaded": False, "neural_training_performed": False, "final_test_accessed": False, "phase3_certified": False}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACQUISITION_COVERAGE_PROTOCOL_V98.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_acquisition_coverage/coverage_v98.json"); args = parser.parse_args()
    root = Path.cwd().resolve(); output = (root / args.output).resolve()
    if output.exists(): raise Phase3Error("acquisition coverage output exists")
    result = run(root, (root / args.protocol).resolve()); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
