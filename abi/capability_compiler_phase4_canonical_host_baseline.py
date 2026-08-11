"""Evaluate the immutable pre-acquisition LayerCake host on the ABI suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from . import capability_compiler_phase3_copy_balanced_transition as balanced
from . import capability_compiler_phase3_qualified_transition_control as qualified
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-canonical-host-baseline/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "SEALED_READ_ONLY_CANONICAL_HOST_BASELINE" or protocol.get("training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("canonical host baseline governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"canonical host baseline binding changed: {relative}")
    base_path = root / protocol["host_protocol"]
    base = _json(base_path)
    if sha256_file(base_path) != protocol["host_protocol_sha256"] or base["host"]["parent_checkpoint_sha256"] != protocol["host_checkpoint_sha256"]:
        raise Phase3Error("canonical host identity changed")
    return protocol, sha256_file(path), base


def evaluate(root: Path, protocol_path: Path, output: Path | None = None) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    if output is not None and output.exists():
        raise Phase3Error("immutable canonical host baseline output exists")
    device = torch.device("cuda")
    _, model, tokenizer, _ = qualified._load_parent(root, base, device)
    model.eval()
    probes = development_probes(root / base["development"]["catalog_path"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / base["development"]["teacher_reference"]).open(encoding="utf-8"))}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        value, tokens, route = balanced._generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device)
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "output": value, "output_token_ids": tokens, "automatic_route": route, "functional_pass_v1": evaluate_functional(value, probe["evaluator"]), "repetition_collapse_v2": repetition_collapse_v2(value)})
        if (index + 1) % 100 == 0:
            print(json.dumps({"canonical_host_evaluated": index + 1}), flush=True)
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passed = sum(row["functional_pass_v1"] for row in values)
        per[capability] = {"passes_v1": passed, "observations": len(values), "wilson_v1": wilson(passed, len(values)), "collapses_v2": sum(row["repetition_collapse_v2"] for row in values)}
    probe_map = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=int(protocol["bootstrap_seed"]))
    threshold = base["absolute_screen"]
    gates = {
        "per_capability_functional": all(value["wilson_v1"]["point"] >= float(threshold["per_capability_functional_point_estimate_minimum"]) and value["wilson_v1"]["lower_95"] >= float(threshold["per_capability_functional_wilson_lower_minimum"]) for value in per.values()),
        "critical_capabilities": all(per[name]["wilson_v1"]["point"] >= float(threshold["critical_point_minimum"]) and per[name]["wilson_v1"]["lower_95"] >= float(threshold["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapse": sum(row["repetition_collapse_v2"] for row in rows) == 0,
        "teacher_noninferior": relative["lower_95"] >= float(base["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "host_checkpoint_immutable": sha256_file(root / base["host"]["parent_path"] / "model.safetensors") == protocol["host_checkpoint_sha256"],
        "final_test_not_accessed": True,
    }
    result = {"format": "abi-capability-compiler-phase4-canonical-host-baseline-result/1", "status": "PASS_CANONICAL_HOST_BASELINE_GATES" if all(gates.values()) else "FAIL_CANONICAL_HOST_BASELINE_GATES", "protocol_sha256": protocol_sha, "host_checkpoint_sha256": protocol["host_checkpoint_sha256"], "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows), "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in rows), "per_capability": per, "teacher_comparison_v1": relative, "gates": gates, "wall_seconds": time.perf_counter() - started, "training_performed": False, "new_teacher_information": 0, "final_test_accessed": False, "phase4_certified": False, "claim_boundary": "Read-only immutable host baseline; no artifact result, frontier, minimum, runtime, matched-baseline, final-test, Phase 4 certificate, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if output is not None:
        output.mkdir(parents=True)
        _write_immutable(output / "development_outputs.jsonl", b"".join(canonical_json_bytes(row) for row in rows))
        _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = evaluate(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
