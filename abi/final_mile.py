"""Fail-closed utilities for ABI's externally gated final-mile campaign.

This module deliberately does not execute model inference or training.  Its
first responsibility is to freeze the exact product that later portability,
human, and external-hardware work is allowed to challenge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


class FinalMileError(RuntimeError):
    """Raised when a final-mile identity or governance check fails."""


STARTING_COMMIT = "a2f2f8f685119c658048faa50b58d01544bc6e92"
LAYERCAKE_STARTING_COMMIT = "662c5a9b7264a1a5478c9dfb656f35c450e2504f"
READINESS_MANIFEST = Path(
    "results/abi_capability_compiler_phase8/readiness_v1073/manifest.json"
)

AUTHORITY_FILES = (
    Path("evidence/current/ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V1089.json"),
    Path("CURRENT_PROJECT_STATUS.md"),
    Path("evidence/current/ABI_CAPABILITY_COMPILER_PHASE7_CERTIFICATE_V1.json"),
    Path("evidence/current/ABI_CAPABILITY_COMPILER_PHASE8_LOCAL_READINESS_RESULT_V1074.json"),
    Path(
        "evidence/current/ABI_CAPABILITY_COMPILER_PHASE8_LOCAL_CLEAN_REHEARSAL_RESULT_V1088.json"
    ),
    Path("ABI_CAPABILITY_COMPILER_PHASE7_PRODUCT_MANIFEST_V1.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE7_RUNTIME_HOST_OVERLAY_V1.json"),
)

COMPARATOR_REGISTRY_FILES = (
    Path("ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_V1.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_REPAIR1_V2.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE4_B40_BASELINE_PACK_PROTOCOL_V987.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE4_B40_BASELINE_HEADLINE_PROTOCOL_V996.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE4_B40_FRONTIER_RESULT_V1015.json"),
)

HUMAN_FILES = (
    Path("ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_PROTOCOL_V1.json"),
    Path("ABI_CAPABILITY_COMPILER_PHASE2_RATER_SESSION_PROTOCOL_V1.json"),
    Path("evidence/current/ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_RATING_READINESS_AUDIT_V554.json"),
    Path("results/abi_capability_compiler_phase2/human_rating_packet_v1/manifest.json"),
)

EXTERNAL_REPRODUCTION_FILES = (
    Path("docs/PHASE8_EXTERNAL_REPRODUCTION_V1.md"),
    Path("PHASE8_EXTERNAL_OPERATOR_ATTESTATION_TEMPLATE_V1.json"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalMileError(f"cannot read required JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FinalMileError(f"required JSON is not an object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalMileError(f"required frozen input is missing: {path}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_binding(repository: Path, commit: str, relative: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None
    payload = completed.stdout
    return {
        "path": f"git:{commit}:{relative}",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _frozen_binding(root: Path, relative: Path, starting_commit: str) -> dict[str, Any]:
    git_value = _git_binding(root, starting_commit, relative.as_posix())
    return git_value if git_value is not None else _binding(root / relative)


def _manifest_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    starting_commit: str,
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != manifest.get("file_count"):
        raise FinalMileError("Phase 8 readiness inventory depth changed")
    inventory: dict[str, Any] = {}
    for relative, declared in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(declared, dict):
            raise FinalMileError("invalid Phase 8 readiness inventory entry")
        path = (root / relative).resolve()
        relocated_from: str | None = None
        actual = _binding(path) if path.is_file() else None
        if (
            actual is not None
            and actual["bytes"] == declared.get("bytes")
            and actual["sha256"] == declared.get("sha256")
        ):
            resolved_path = (
                path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else path.as_posix()
            )
            resolved_from_git = False
        elif not relative.startswith("../"):
            candidates = [
                candidate.resolve()
                for candidate in (root / "evidence" / "current").rglob(Path(relative).name)
                if candidate.is_file() and sha256_file(candidate) == declared.get("sha256")
            ]
            if len(candidates) == 1:
                relocated_from = relative
                path = candidates[0]
                actual = _binding(path)
                resolved_path = path.relative_to(root).as_posix()
                resolved_from_git = False
            elif declared.get("tracked") is True:
                actual = _git_binding(root, starting_commit, relative)
                resolved_path = f"git:{starting_commit}:{relative}"
                resolved_from_git = True
        elif declared.get("tracked") is True:
            layercake = (root.parent / "layercake_release").resolve()
            repository_relative = relative.removeprefix("../layercake_release/")
            actual = _git_binding(layercake, LAYERCAKE_STARTING_COMMIT, repository_relative)
            resolved_path = f"git:{LAYERCAKE_STARTING_COMMIT}:{repository_relative}"
            resolved_from_git = True
        if actual is None:
            raise FinalMileError(f"Phase 8 inventory binding is unavailable: {relative}")
        if (
            actual["bytes"] != declared.get("bytes")
            or actual["sha256"] != declared.get("sha256")
        ):
            raise FinalMileError(f"Phase 8 inventory binding changed: {relative}")
        inventory[relative] = {
            "repository": declared.get("repository"),
            "tracked_at_start": bool(declared.get("tracked")),
            "resolved_path": resolved_path,
            "relocated_in_curated_tree": relocated_from is not None,
            "resolved_from_frozen_git": resolved_from_git,
            "bytes": actual["bytes"],
            "sha256": actual["sha256"],
        }
    return inventory


def freeze_starting_point(
    root: Path,
    *,
    output: Path,
    starting_commit: str = STARTING_COMMIT,
    clean_at_campaign_start: bool = True,
) -> dict[str, Any]:
    """Freeze the already-certified product without changing any artifact."""
    root = root.resolve()
    manifest_path = (root / READINESS_MANIFEST).resolve()
    manifest = _read_object(manifest_path)
    if (
        manifest.get("format")
        != "abi-capability-compiler-phase8-release-readiness-result/1"
        or manifest.get("status") != "PASS_PHASE8_LOCAL_RELEASE_READINESS"
        or manifest.get("phase8_certified") is not False
    ):
        raise FinalMileError("Phase 8 readiness authority changed")
    if not _git(root, "cat-file", "-e", f"{starting_commit}^{{commit}}") == "":
        raise FinalMileError("starting commit check returned unexpected output")

    authority = {
        path.as_posix(): _frozen_binding(root, path, starting_commit)
        for path in AUTHORITY_FILES
    }
    comparators = {
        path.as_posix(): _frozen_binding(root, path, starting_commit)
        for path in COMPARATOR_REGISTRY_FILES
    }
    human = {
        path.as_posix(): _frozen_binding(root, path, starting_commit) for path in HUMAN_FILES
    }
    external = {
        path.as_posix(): _frozen_binding(root, path, starting_commit)
        for path in EXTERNAL_REPRODUCTION_FILES
    }
    inventory = _manifest_inventory(root, manifest, starting_commit=starting_commit)

    package_paths = [
        relative
        for relative in inventory
        if relative.endswith(".cake") or relative.endswith(".pub")
    ]
    product_manifest = _read_object(root / "ABI_CAPABILITY_COMPILER_PHASE7_PRODUCT_MANIFEST_V1.json")
    phase7 = _read_object(
        root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE7_CERTIFICATE_V1.json"
    )
    state = _read_object(
        root / "evidence/current/ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V1089.json"
    )
    result: dict[str, Any] = {
        "format": "abi-final-mile-frozen-starting-point/1",
        "status": "FROZEN_EXISTING_SAME_MACHINE_PRODUCT_FOR_FINAL_MILE_CHALLENGE",
        "repository": "https://github.com/Yoder23/abi",
        "repository_commit": starting_commit,
        "repository_head_when_freeze_generated": _git(root, "rev-parse", "HEAD"),
        "working_tree_clean_at_campaign_start": clean_at_campaign_start,
        "historical_evidence_changed": False,
        "authority": authority,
        "phase8_readiness_manifest": {
            **_binding(manifest_path),
            "declared_evidence_sha256": manifest.get("evidence_sha256"),
            "inventory_files": len(inventory),
            "inventory_bytes": sum(row["bytes"] for row in inventory.values()),
        },
        "release_inventory": inventory,
        "package_bindings": {path: inventory[path] for path in package_paths},
        "comparator_registry": comparators,
        "human_rating_packet": human,
        "external_reproduction_instructions": external,
        "runtime_bindings": {
            path: row
            for path, row in inventory.items()
            if path.endswith(".py") and ("runtime" in path or path.startswith("../layercake_release"))
        },
        "evaluator_and_data_bindings": {
            path: row
            for path, row in inventory.items()
            if path.startswith("catalogs/")
            or path.endswith("outputs.jsonl")
            or "/verify_" in path
            or path.endswith("_verify.py")
        },
        "current_gate_statuses": state["phase_status"],
        "certified_same_machine_results": phase7["certified_results"],
        "product_identity": product_manifest,
        "current_claim_ceiling": {
            "tier_a_same_machine_integrated_product": "CERTIFIED_BOUNDED",
            "tier_b_cross_host_abi_portability": "NOT_TESTED_ACROSS_DISTINCT_ARCHITECTURE_FAMILIES",
            "tier_c_independent_external_reproduction": "BLOCKED_EXTERNAL_HARDWARE_AND_OPERATOR",
            "tier_d_full_abi_moonshot": "NOT_PROVEN",
            "phase2_human_preferences": "0/21000",
            "global_minimum_claimed": False,
            "universal_portability_claimed": False,
        },
        "mutation_rule": (
            "Any changed artifact is a new version; every dependent result must be rerun."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if output.exists():
        raise FinalMileError(f"immutable freeze output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--starting-commit", default=STARTING_COMMIT)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = freeze_starting_point(
        root,
        output=(root / args.output).resolve(),
        starting_commit=args.starting_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
