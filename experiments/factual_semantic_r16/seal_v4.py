"""Final local R16 seal with repaired rendered-prompt accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .accounting_v2 import account_v2
from .hostile_audit_v3 import audit_v3
from .verify_strict_v3 import verify_v3


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def seal_v4(
    config: Path,
    reveal: Path,
    run_dir: Path,
    strict_path: Path,
    live_path: Path,
    hostile_path: Path,
    accounting_v1_path: Path,
    certificate_v3_path: Path,
    strict_v3_path: Path,
    hostile_v3_path: Path,
    accounting_v2_path: Path,
    public_requalification: Path,
) -> dict[str, Any]:
    prior = _verified(certificate_v3_path, "certificate-v3")
    if (
        prior["evidence"]["strict_v2_sha256"] != sha256_file(strict_path)
        or prior["evidence"]["hostile_v2_sha256"] != sha256_file(hostile_path)
        or prior["evidence"]["accounting_sha256"] != sha256_file(accounting_v1_path)
    ):
        raise R14Error("R16 certificate-v3 dependency changed")
    strict = _verified(strict_v3_path, "strict-v3")
    recomputed_strict = verify_v3(
        config, reveal, run_dir, live_path, public_requalification
    )
    if strict != recomputed_strict:
        raise R14Error("R16 strict-v3 changed")
    hostile = _verified(hostile_v3_path, "hostile-v3")
    recomputed_hostile = audit_v3(
        config, reveal, run_dir, live_path, public_requalification
    )
    if hostile != recomputed_hostile:
        raise R14Error("R16 hostile-v3 changed")
    accounting = _verified(accounting_v2_path, "accounting-v2")
    if accounting != account_v2(run_dir):
        raise R14Error("R16 accounting-v2 changed")
    certificate = {
        **{key: value for key, value in prior.items() if key != "evidence_sha256"},
        "format": "abi-r16-bounded-certificate/4",
        "repair_of": "results/factual_semantic_r16/heldout_v1_certificate_v3.json",
        "repair_reason": (
            "bind rendered-prompt accounting, actual live files, direct source snapshot "
            "rehash, candidate disclosure, and public protocol requalification"
        ),
        "information_accounting": accounting,
        "results": {
            **prior["results"],
            "hostile_cases_rejected": hostile["cases_passed"],
            "actual_live_files_rehashed": strict["actual_live_files_rehashed"],
            "physical_extraction_trees_verified": strict[
                "physical_extraction_trees_verified"
            ],
            "source_snapshot_files_rehashed": strict["source_snapshot_files_rehashed"],
            "public_requalification_artifacts_verified": strict[
                "public_requalification_artifacts_verified"
            ],
            "correct_answer_present_once_in_candidate_rows": strict[
                "correct_answer_present_once_in_candidate_rows"
            ],
            "explicit_labeled_answer_fields_in_compiler_bundle": strict[
                "explicit_labeled_answer_fields_in_compiler_bundle"
            ],
        },
        "candidate_boundary": strict["candidate_boundary"],
        "historical_public_protocol_declared_sha256": strict[
            "historical_public_protocol_declared_sha256"
        ],
        "historical_public_protocol_matches_current_file": strict[
            "historical_public_protocol_matches_current_file"
        ],
        "public_requalification_status": "PASS_POST_REVEAL_ASSURANCE_ONLY",
        "evidence": {
            **prior["evidence"],
            "certificate_v3_sha256": sha256_file(certificate_v3_path),
            "strict_v3_sha256": sha256_file(strict_v3_path),
            "hostile_v3_sha256": sha256_file(hostile_v3_path),
            "accounting_v2_sha256": sha256_file(accounting_v2_path),
            "public_requalification_sha256": sha256_file(
                public_requalification / "receipt.json"
            ),
        },
    }
    certificate["evidence_sha256"] = evidence_hash(certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--hostile", type=Path, required=True)
    parser.add_argument("--accounting-v1", type=Path, required=True)
    parser.add_argument("--certificate-v3", type=Path, required=True)
    parser.add_argument("--strict-v3", type=Path, required=True)
    parser.add_argument("--hostile-v3", type=Path, required=True)
    parser.add_argument("--accounting-v2", type=Path, required=True)
    parser.add_argument("--public-requalification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal_v4(
        args.config,
        args.reveal,
        args.run_dir,
        args.strict,
        args.live,
        args.hostile,
        args.accounting_v1,
        args.certificate_v3,
        args.strict_v3,
        args.hostile_v3,
        args.accounting_v2,
        args.public_requalification,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
