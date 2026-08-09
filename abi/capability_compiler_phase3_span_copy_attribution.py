"""No-model attribution of unseen target pieces to repeated source spans."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_route_bridge import _select_controls
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_unicode_span_copy_feasibility import _targeted_rows


FORMAT = "abi-capability-compiler-phase3-span-copy-attribution/1"
IDENTITY = re.compile(rb"^[A-Za-z0-9_]+$")


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("span-copy attribution governance changed")
    for relative, expected in protocol["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"span-copy attribution binding changed: {relative}")
    layercake = (root / protocol["layercake_host"]["repository"]).resolve(); sys.path.insert(0, str(layercake))
    from layercake_extensions.selective_boundary_bpe_direct_neural_core import SelectiveBoundaryBpeTokenizer
    tokenizer = SelectiveBoundaryBpeTokenizer(json.loads((root / protocol["tokenizer"]).read_text(encoding="utf-8")))
    original = load_phase1_ir(root / protocol["phase1_ir"]); targeted = _targeted_rows(root / protocol["targeted_ir"]); controls = _select_controls(original, tokenizer); control_by_capability = {capability: controls[index][1] for index, capability in enumerate(CAPABILITIES)}
    seen = {capability: set() for capability in CAPABILITIES}
    for row in original + targeted:
        seen[str(row["capability"])].update(tokenizer.split(str(row["normalized_output"])))
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    per = {capability: Counter() for capability in CAPABILITIES}
    for probe in development_probes(root / protocol["development_catalog"]):
        capability = str(probe["canonical_capability"]); body = _semantic_segments(str(probe["prompt"]))[-1]; source = [control_by_capability[capability]] + tokenizer.split("\n" + body); counts = Counter(source)
        for piece in tokenizer.split(str(teacher[str(probe["probe_id"])]["output"])):
            if piece in seen[capability]: continue
            report = per[capability]; report["unseen_actions"] += 1; occurrences = counts[piece]
            if IDENTITY.fullmatch(piece) is None: report["ineligible_actions"] += 1
            elif occurrences == 1: report["unique_pointerable_actions"] += 1
            elif occurrences > 1: report["repeated_span_pointerable_actions"] += 1
            else: report["absent_actions"] += 1
    summaries = {}
    for capability, counts in per.items():
        unseen = counts["unseen_actions"]; current = counts["unique_pointerable_actions"]; span = current + counts["repeated_span_pointerable_actions"]
        summaries[capability] = {**dict(counts), "current_pointerable_ratio": current / unseen if unseen else 1.0, "span_pointerable_ratio": span / unseen if unseen else 1.0, "span_increment": counts["repeated_span_pointerable_actions"] / unseen if unseen else 0.0}
    critical = ("coherence", "fact_free_reasoning")
    gates = {"critical_span_pointerable_ratio": all(summaries[value]["span_pointerable_ratio"] >= float(protocol["pass_gates"]["critical_span_pointerable_ratio_minimum"]) for value in critical), "critical_span_increment": all(summaries[value]["span_increment"] >= float(protocol["pass_gates"]["critical_span_increment_minimum"]) for value in critical), "critical_repeated_actions": all(summaries[value].get("repeated_span_pointerable_actions", 0) >= int(protocol["pass_gates"]["critical_repeated_actions_minimum"]) for value in critical)}
    return {"format": "abi-capability-compiler-phase3-span-copy-attribution-result/1", "status": "PASS_SPAN_COPY_ATTRIBUTION" if all(gates.values()) else "FAIL_SPAN_COPY_ATTRIBUTION", "per_capability": summaries, "gates": gates, "teacher_model_loaded": False, "neural_training_performed": False, "layercake_host_changed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "No-model attribution only; a pass authorizes only a separate span-action host feasibility and construct."}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("span-copy attribution output exists")
    result = execute(root, root / args.protocol); output.parent.mkdir(parents=True, exist_ok=True); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps({"status": result["status"], "critical": {key: result["per_capability"][key] for key in ("coherence", "fact_free_reasoning")}}, indent=2)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
