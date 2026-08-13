"""Read-only paired audit on the frozen Phase 4 metamorphic coherence suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase4_abi_lineage as lineage
from . import capability_compiler_phase4_capability_isolated_adaptation as adapted
from . import capability_compiler_phase4_functional_validation as functional
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson


FORMAT = "abi-capability-compiler-phase4-metamorphic-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return rows


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_METAMORPHIC_COHERENCE_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("promotion_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("metamorphic audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"metamorphic audit binding changed: {relative}")
    lineage_protocol, _ = lineage.load_protocol(root, root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _binomial_one_sided(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(wins, discordant + 1)) / (2**discordant)


def _paired_prompt_bootstrap(old: Mapping[str, bool], new: Mapping[str, bool], rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, float]:
    differences = [int(new[str(row["ir_record_id"])]) - int(old[str(row["ir_record_id"])]) for row in rows]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    return {
        "point": sum(differences) / len(differences),
        "lower_95": estimates[int(0.025 * (samples - 1))],
        "upper_95": estimates[int(0.975 * (samples - 1))],
    }


def _strata(rows: list[dict[str, Any]], old: Mapping[str, bool], new: Mapping[str, bool], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        selected = [row for row in rows if str(row[key]) == value]
        identifiers = [str(row["ir_record_id"]) for row in selected]
        inherited = sum(old[identifier] for identifier in identifiers)
        adapted_count = sum(new[identifier] for identifier in identifiers)
        result[value] = {
            "observations": len(identifiers),
            "inherited_passes": inherited,
            "adapted_passes": adapted_count,
            "paired_difference": (adapted_count - inherited) / len(identifiers),
        }
    return result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    suite = root / protocol["suite"]
    rows = _jsonl(suite)
    manifest = _json(root / protocol["suite_manifest"])
    gates = {
        "suite_build_passed": manifest["status"] == "PASS_MODEL_BLIND_METAMORPHIC_SUITE_BUILD",
        "suite_hash_exact": manifest["suite_sha256"] == sha256_file(suite),
        "expected_depth": len(rows) == int(protocol["expected_records"]),
        "ten_namespaces": len({row["namespace"] for row in rows}) == 10,
        "four_families": len({row["family"] for row in rows}) == 4,
        "no_training_rows": not any(row["training_eligible"] for row in rows),
        "no_teacher_outputs": not any(row["teacher_output_present"] for row in rows),
    }
    return {
        "status": "PASS_METAMORPHIC_AUDIT_PREFLIGHT" if all(gates.values()) else "FAIL_METAMORPHIC_AUDIT_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "records_per_system": len(rows),
        "systems": ["inherited", "adapted"],
        "gates": gates,
        "training_performed": False,
        "promotion_authorized": False,
        "final_test_accessed": False,
    }


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable output exists or CUDA unavailable")
    suite_path = root / protocol["suite"]
    rows = _jsonl(suite_path)
    device = torch.device("cuda")
    run = protocol["runs"][0]
    run_dir = root / run["lineage_dir"]
    model, tokenizer, _, _, _ = adapted._load_components(root, protocol, lineage_protocol, run, device)
    inherited = isolated.RouteIsolatedResidual().to(device)
    inherited.load_state_dict(load_file(str(run_dir / "v526" / "control_bridge.safetensors"), device="cuda"), strict=True)
    inherited.eval()
    trained = adapted.CapabilityIsolatedResidual().to(device)
    trained.load_state_dict(load_file(str(root / protocol["adapted_checkpoint"]), device="cuda"), strict=True)
    trained.eval()
    old, inherited_rows = functional._guarded_generate(model, tokenizer, inherited, rows, run_dir, protocol, device, system="inherited")
    new, adapted_rows = functional._guarded_generate(model, tokenizer, trained, rows, run_dir, protocol, device, system="adapted")
    attributes = {str(row["ir_record_id"]): row for row in rows}
    for evidence in inherited_rows + adapted_rows:
        source = attributes[str(evidence["record_id"])]
        evidence["namespace"] = source["namespace"]
        evidence["family"] = source["family"]
    identifiers = [str(row["ir_record_id"]) for row in rows]
    old_total = sum(old[identifier] for identifier in identifiers)
    new_total = sum(new[identifier] for identifier in identifiers)
    interval = wilson(new_total, len(rows))
    wins = sum(new[identifier] and not old[identifier] for identifier in identifiers)
    losses = sum(old[identifier] and not new[identifier] for identifier in identifiers)
    namespaces = _strata(rows, old, new, "namespace")
    families = _strata(rows, old, new, "family")
    bootstrap = _paired_prompt_bootstrap(old, new, rows, samples=int(protocol["paired_bootstrap"]["samples"]), seed=int(protocol["paired_bootstrap"]["seed"]))
    namespace_wins = sum(value["paired_difference"] > 0 for value in namespaces.values())
    minimum_delta = min(value["paired_difference"] for value in namespaces.values())
    minimum_family_delta = min(value["paired_difference"] for value in families.values())
    gates = {
        "adapted_strictly_improves": new_total > old_total,
        "adapted_point": interval["point"] >= float(protocol["thresholds"]["point"]),
        "adapted_lower_95": interval["lower_95"] >= float(protocol["thresholds"]["lower_95"]),
        "paired_bootstrap_lower_positive": bootstrap["lower_95"] > 0,
        "paired_mcnemar_one_sided": _binomial_one_sided(wins, losses) <= float(protocol["thresholds"]["paired_p_maximum"]),
        "namespace_breadth": namespace_wins >= int(protocol["thresholds"]["minimum_strict_namespace_wins"]),
        "namespace_regression_bounded": minimum_delta >= -float(protocol["thresholds"]["maximum_stratum_regression"]),
        "family_regression_bounded": minimum_family_delta >= -float(protocol["thresholds"]["maximum_stratum_regression"]),
        "zero_collapse_both": sum(row["repetition_collapse_v2"] for row in inherited_rows + adapted_rows) == 0,
        "no_training": True,
        "no_promotion": True,
        "final_test_not_accessed": True,
    }
    raw = output.parent / "outputs.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in inherited_rows + adapted_rows))
    result = {
        "format": "abi-capability-compiler-phase4-metamorphic-audit-result/1",
        "status": "PASS_METAMORPHIC_COHERENCE_VALIDATION_DISTINGUISHES_ADAPTED_ROUTE" if all(gates.values()) else "FAIL_METAMORPHIC_COHERENCE_VALIDATION",
        "protocol_sha256": protocol_sha,
        "suite_sha256": sha256_file(suite_path),
        "observations_per_system": len(rows),
        "inherited_passes": old_total,
        "adapted_passes": new_total,
        "adapted_wilson": interval,
        "discordance": {"adapted_only_passes": wins, "inherited_only_passes": losses, "mcnemar_one_sided_p": _binomial_one_sided(wins, losses)},
        "paired_prompt_bootstrap": bootstrap,
        "by_namespace": namespaces,
        "by_family": families,
        "gates": gates,
        "outputs_sha256": sha256_file(raw),
        "training_performed": False,
        "promotion_authorized": False,
        "final_test_accessed": False,
        "interpretation": "A pass supports only a separately preregistered no-training coherence-route construction and development screen. It does not construct or promote a candidate.",
        "claim_boundary": "Prospective read-only validation audit; no candidate construction, stable frontier, matched baseline, final test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("audit")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else audit(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
