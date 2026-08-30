"""Freeze a repaired technical-proof commit into an immutable release candidate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .final_validation import CAPABILITY_PATHS, evidence_hash
from .strict_validation import read_json, sha256_file, verify

TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RepairedCandidateError(RuntimeError):
    """Raised when the technical proof cannot be frozen exactly."""


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RepairedCandidateError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def build(root: Path, *, commit: str, tag: str) -> dict[str, Any]:
    root = root.resolve()
    if not COMMIT_RE.fullmatch(commit) or not TAG_RE.fullmatch(tag):
        raise RepairedCandidateError("malformed technical-proof commit or tag")
    if _git(root, "rev-parse", commit) != commit:
        raise RepairedCandidateError("technical-proof commit is unavailable")
    if _git(root, "rev-list", "-n", "1", tag) != commit:
        raise RepairedCandidateError("technical-proof tag does not resolve to commit")
    if _git(root, "status", "--short"):
        raise RepairedCandidateError("technical-proof worktree is not clean")
    strict = verify(root)
    strict_path = root / (
        "results/abi_final_validation_v2/strict_validation_r5_recursive_bound.json"
    )
    recorded = read_json(strict_path)
    if recorded.get("evidence_sha256") != evidence_hash(strict):
        raise RepairedCandidateError("recorded strict certificate is stale")
    hostile_path = root / (
        "results/abi_final_validation_v2/strict_hostile_pre_public_r5.json"
    )
    hostile = read_json(hostile_path)
    if (
        hostile.get("format") != "abi-v2-strict-hostile-verification/3"
        or hostile.get("status") != "PASS_STRICT_VERIFIER_FAILS_CLOSED"
        or hostile.get("mutations_rejected") != hostile.get("mutations_required")
        or hostile.get("strict_verifier_source_sha256")
        != sha256_file(root / "abi_v2/strict_validation.py")
        or hostile.get("hostile_verifier_source_sha256")
        != sha256_file(root / "abi_v2/strict_hostile.py")
    ):
        raise RepairedCandidateError("strict hostile receipt is incomplete")
    value: dict[str, Any] = {
        "format": "abi-v2-repaired-frozen-release-candidate/5",
        "status": "TECHNICAL_PROOF_FROZEN_AWAITING_PUBLIC_RECONSTRUCTION_AND_RED_TEAM",
        "repository": "https://github.com/Yoder23/abi",
        "technical_proof_commit": commit,
        "technical_proof_tag": tag,
        "strict_certificate": {
            "path": strict_path.relative_to(root).as_posix(),
            "bytes": strict_path.stat().st_size,
            "sha256": sha256_file(strict_path),
            "evidence_sha256": recorded["evidence_sha256"],
            "required_inputs_aggregate_sha256": recorded["required_inputs"][
                "aggregate_sha256"
            ],
        },
        "strict_hostile_receipt": {
            "path": hostile_path.relative_to(root).as_posix(),
            "bytes": hostile_path.stat().st_size,
            "sha256": sha256_file(hostile_path),
            "evidence_sha256": hostile["evidence_sha256"],
            "mutations_rejected": hostile["mutations_rejected"],
        },
        "capability_artifacts": {
            capability: {
                "path": relative,
                "bytes": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
            for capability, relative in CAPABILITY_PATHS.items()
        },
        "external_gates": {
            "public_hash_addressed_assets": "PENDING",
            "fresh_public_reconstruction": "PENDING",
            "fresh_blind_codex_red_team": "PENDING",
            "human_ratings": "CLOSED",
            "independent_hardware": "CLOSED",
        },
        "trusted_scientific_booleans_consumed": 0,
    }
    value["evidence_sha256"] = evidence_hash(value)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise RepairedCandidateError(f"immutable candidate exists: {output}")
    value = build(root, commit=args.commit, tag=args.tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
