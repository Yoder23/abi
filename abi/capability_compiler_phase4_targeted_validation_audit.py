"""Read-only targeted coherence validation audit for the frozen hard-seed routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase4_abi_lineage as lineage
from . import capability_compiler_phase4_capability_isolated_adaptation as adapted
from . import capability_compiler_phase4_functional_validation as functional
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson


FORMAT = "abi-capability-compiler-phase4-targeted-validation-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_TARGETED_COHERENCE_DISCRIMINATION" or protocol.get("training_authorized") is not False or protocol.get("promotion_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("targeted validation audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"targeted validation binding changed: {relative}")
    lineage_protocol, _ = lineage.load_protocol(root, root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _rank(record_id: str, salt: str) -> str:
    return hashlib.sha256((salt + "\0" + record_id).encode()).hexdigest()


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    selected, _ = lineage._selected_rows(root, lineage_protocol, _json(root / lineage_protocol["budget_manifest"]), protocol["budget"])
    targeted = [row for row in selected["v138_targeted_ir"] if row["capability"] == "coherence"]
    host = [row for row in selected["v480_host_supervision"] if row["capability"] == "coherence"]
    count = int(protocol["records_per_source"])
    if len(targeted) < count or len(host) < count: raise Phase3Error("targeted coherence evidence lacks registered depth")
    if any(not isinstance(row.get("functional_evaluator"), dict) for row in targeted + host): raise Phase3Error("targeted evaluator missing")
    return {"status": "PASS_TARGETED_VALIDATION_AUDIT_PREFLIGHT", "protocol_sha256": protocol_sha, "available": {"targeted_ir": len(targeted), "host_supervision": len(host)}, "selected_per_source": count, "total_per_system": count * 2, "systems": ["inherited", "adapted"], "training_performed": False, "promotion_authorized": False, "final_test_accessed": False}


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("immutable output exists or CUDA unavailable")
    selected, _ = lineage._selected_rows(root, lineage_protocol, _json(root / lineage_protocol["budget_manifest"]), protocol["budget"])
    count = int(protocol["records_per_source"]); salt = str(protocol["selection_salt"])
    targeted = sorted((row for row in selected["v138_targeted_ir"] if row["capability"] == "coherence"), key=lambda row: _rank(str(row["ir_record_id"]), salt))[:count]
    catalog = {row["probe_id"]: row for row in _json(root / protocol["targeted_catalog"])["probes"]}
    host = sorted((row for row in selected["v480_host_supervision"] if row["capability"] == "coherence"), key=lambda row: _rank(str(row["record_id"]), salt))[:count]
    normalized = [dict(row, validation_source="targeted_ir") for row in targeted]
    for row in host:
        probe = catalog[str(row["probe_id"])]
        normalized.append({"ir_record_id": str(row["record_id"]), "capability": "coherence", "normalized_generation_prompt": str(row["host_prompt"]), "generation_max_new_tokens": int(probe["max_new_tokens"]), "functional_evaluator": row["functional_evaluator"], "validation_source": "host_supervision"})
    device = torch.device("cuda"); run = protocol["runs"][0]; run_dir = root / run["lineage_dir"]
    model, tokenizer, _, _, _ = adapted._load_components(root, protocol, lineage_protocol, run, device)
    inherited = isolated.RouteIsolatedResidual().to(device); inherited.load_state_dict(load_file(str(run_dir / "v526" / "control_bridge.safetensors"), device="cuda"), strict=True); inherited.eval()
    trained = adapted.CapabilityIsolatedResidual().to(device); trained.load_state_dict(load_file(str(root / protocol["adapted_checkpoint"]), device="cuda"), strict=True); trained.eval()
    inherited_pass, inherited_rows = functional._guarded_generate(model, tokenizer, inherited, normalized, run_dir, protocol, device, system="inherited")
    adapted_pass, adapted_rows = functional._guarded_generate(model, tokenizer, trained, normalized, run_dir, protocol, device, system="adapted")
    by_source = {}
    for source in ("targeted_ir", "host_supervision"):
        ids = [str(row["ir_record_id"]) for row in normalized if row["validation_source"] == source]
        by_source[source] = {"observations": len(ids), "inherited_passes": sum(inherited_pass[key] for key in ids), "adapted_passes": sum(adapted_pass[key] for key in ids)}
    ids = [str(row["ir_record_id"]) for row in normalized]; old = sum(inherited_pass[key] for key in ids); new = sum(adapted_pass[key] for key in ids); interval = wilson(new, len(ids))
    gates = {"adapted_strictly_improves": new > old, "adapted_point": interval["point"] >= float(protocol["thresholds"]["point"]), "adapted_lower_95": interval["lower_95"] >= float(protocol["thresholds"]["lower_95"]), "zero_collapse_both": sum(row["repetition_collapse_v2"] for row in inherited_rows + adapted_rows) == 0, "no_training": True, "no_promotion": True, "final_test_not_accessed": True}
    raw = output.parent / "outputs.jsonl"; output.parent.mkdir(parents=True); _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in inherited_rows + adapted_rows))
    result = {"format": "abi-capability-compiler-phase4-targeted-validation-audit-result/1", "status": "PASS_TARGETED_COHERENCE_VALIDATION_DISTINGUISHES_ADAPTED_ROUTE" if all(gates.values()) else "FAIL_TARGETED_COHERENCE_VALIDATION_NOT_DISCRIMINATIVE", "protocol_sha256": protocol_sha, "observations_per_system": len(ids), "inherited_passes": old, "adapted_passes": new, "adapted_wilson": interval, "by_source": by_source, "gates": gates, "outputs_sha256": sha256_file(raw), "training_performed": False, "promotion_authorized": False, "final_test_accessed": False, "interpretation": "A pass supports one separately preregistered construction that replaces only the coherence route in the already frozen V679 selected artifact. It does not itself construct or promote a candidate.", "claim_boundary": "Read-only frozen-artifact discrimination audit; no candidate construction, stable frontier, matched baseline, final test, Phase 4 certificate, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); sub=parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); p=sub.add_parser("audit"); p.add_argument("--output", required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); result=preflight(root, root/args.protocol) if args.command=="preflight" else audit(root, root/args.protocol, root/args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
