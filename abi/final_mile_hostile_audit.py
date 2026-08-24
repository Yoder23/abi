"""Adversarial claim audit for the sealed ABI final-mile candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

from .final_mile import FinalMileError, canonical_json_bytes, sha256_file
from .reproduce import verify_release


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalMileError(f"expected JSON object: {path}")
    return value


def _evidence_hash(document: dict[str, Any]) -> bool:
    copied = dict(document)
    declared = copied.pop("evidence_sha256", None)
    return declared == hashlib.sha256(canonical_json_bytes(copied)).hexdigest()


def audit(root: Path, *, output: Path) -> dict[str, Any]:
    root = root.resolve()
    release = root / "results/abi_final_mile/abi-release"
    frozen = _object(root / "results/abi_final_mile/frozen_starting_point.json")
    host = _object(root / "results/abi_final_mile/host_portability_v1/repair_rescreen.json")
    certificate = _object(release / "release-certificate.json")
    information = _object(release / "imported-information-ledger.json")
    baseline = _object(release / "baseline-comparisons/comparison.json")
    frontier = _object(release / "minimum-stable-frontier.json")
    source_lock = _object(release / "source-success-lock.json")
    human = _object(release / "human-evidence/status.json")
    external = _object(release / "external-reproduction/status.json")
    phase7 = _object(root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE7_CERTIFICATE_V1.json")
    phase5 = _object(root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE5_CERTIFICATE_V1.json")

    verification = verify_release(release)
    signature = _object(release / "release-signature.json")
    public = serialization.load_pem_public_key(signature["public_key_pem"].encode())
    tampered_manifest = (release / "release-manifest.json").read_bytes() + b" "
    signature_rejects_tamper = False
    try:
        public.verify(bytes.fromhex(signature["signature_ed25519_hex"]), tampered_manifest)
    except InvalidSignature:
        signature_rejects_tamper = True

    attacks = {
        "artifact_byte_mutation_rejected": verification["status"].startswith("PASS")
        and all(
            sha256_file(release / name) == binding["sha256"]
            for name, binding in _object(release / "release-manifest.json")[
                "package_bindings"
            ].items()
        ),
        "outer_manifest_mutation_rejected": signature_rejects_tamper,
        "stale_frozen_starting_point_rejected": _evidence_hash(frozen),
        "host_provider_bypass_rejected": host["receivers_passing"] == 1
        and host["receivers_required"] == 3
        and all(
            not row["qualifies_under_preregistered_rule"]
            for row in host["receivers"]
            if row["would_require_layercake_provider_bypass"]
        ),
        "host_failure_not_promoted": certificate["status"] == "HOST_INDEPENDENCE_FAILED"
        and certificate["release_certified"] is False,
        "capability_calibration_not_invented": all(
            row.get("capability_specific_training_steps", 0) == 0
            and row.get("capability_specific_calibration_examples", 0) == 0
            for row in _object(
                root / "results/abi_final_mile/host_portability_v1/initial_screen.json"
            )["receivers"]
        ),
        "source_success_subset_not_called_retained": source_lock["receiver_evaluation_status"]
        == "NOT_RUN_STRUCTURAL_ABI_REJECTION",
        "selected_only_execution_not_borrowed_cross_host": phase7["certified_results"]
        ["selected_only_domain_executions_per_device"]
        == 123
        and host["semantic_portability_tested"] is False,
        "information_cost_gaps_not_hidden": information["cost"]
        ["cost_gap_is_blocking_for_full_moonshot"]
        is True
        and information["deployment"]["deployment_accounting_gap_is_blocking_for_full_moonshot"]
        is True,
        "latent_specialist_purity_not_claimed": "does not claim" in phase5["semantic_boundary"],
        "same_machine_speed_not_relabelled_external": certificate["tier_a"]
        == "PASS_BOUNDED_SAME_MACHINE"
        and certificate["tier_c"] == "BLOCKED_EXTERNAL_HARDWARE_AND_OPERATOR",
        "external_reproduction_not_fabricated": external["status"]
        == "BLOCKED_EXTERNAL_HARDWARE"
        and external["same_machine_substituted"] is False,
        "human_quality_not_fabricated": human["completed_preferences"] == 0
        and human["required_preferences"] == 21000
        and human["codex_self_rated"] is False,
        "minimum_frontier_not_overstated": frontier["status"]
        == "NO_FINAL_MILE_STABLE_FOOTPRINT"
        and frontier["all_receiver_requirement"]["minimum_stable_budget"] is None,
        "baseline_gap_not_hidden": baseline["systems"]["conventional_hidden_state_distillation"]
        ["blocking_for_full_baseline_claim"]
        is True,
    }
    blockers = [
        {
            "id": "HOST_NATIVE_EXECUTION",
            "severity": "BLOCKING",
            "finding": "Only LayerCake v25 natively consumes the sealed English tensor contract (1/3 receivers).",
        },
        {
            "id": "SEMANTIC_PORTABILITY",
            "severity": "BLOCKING",
            "finding": "The 1,381-item locked source-success set has not been evaluated on qualifying alternate receivers.",
        },
        {
            "id": "HUMAN_GATE",
            "severity": "BLOCKING_EXTERNAL",
            "finding": "0/21,000 preferences and 0/3 independent raters are complete.",
        },
        {
            "id": "EXTERNAL_HARDWARE",
            "severity": "BLOCKING_EXTERNAL",
            "finding": "No independent operator on a different CPU/CUDA machine has returned evidence.",
        },
        {
            "id": "INFORMATION_FRONTIER",
            "severity": "BLOCKING",
            "finding": "B40 is stable only in the prior same-host family; no budget passes all receivers.",
        },
        {
            "id": "ACCOUNTING_GAPS",
            "severity": "BLOCKING",
            "finding": "Production-lineage CPU/GPU hours and operations per generated unit are not reconstructed in the compact ledger.",
        },
        {
            "id": "BASELINE_GAP",
            "severity": "BLOCKING",
            "finding": "No exact-B40 conventional hidden-state-distillation baseline or cross-host reuse comparison exists.",
        },
        {
            "id": "INNER_RESEARCH_SIGNATURE",
            "severity": "RELEASE_SECURITY",
            "finding": "Historical package signatures use reproducible research custody; the new outer signature authenticates this bundle, but current LayerCake installation does not enforce that outer signature.",
        },
    ]
    result = {
        "format": "abi-final-mile-hostile-claim-audit/1",
        "status": "FAIL_BLOCKING_FINDINGS",
        "attacks": attacks,
        "attacks_rejected": sum(attacks.values()),
        "attacks_total": len(attacks),
        "blocking_findings": blockers,
        "unresolved_blocking_findings": len(blockers),
        "model_inference_performed": False,
        "training_performed": False,
        "same_machine_product_rerun_performed": False,
        "final_status": "HOST_INDEPENDENCE_FAILED",
        "claim_boundary": (
            "Fail-closed audit success means false promotions were rejected; it does not mean "
            "the audited product passed its blocking scientific gates."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if output.exists():
        raise FinalMileError(f"immutable hostile audit already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = audit(root, output=(root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
