"""Verify the Phase 0 certificate against its immutable implementation commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


def _git_show(repository_root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.resolve().as_posix()}",
            "show",
            f"{commit}:{path}",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot read {path} from implementation commit {commit}")
    return completed.stdout


def validate_certificate(
    certificate: Mapping[str, object],
    repository_root: Path,
) -> list[str]:
    errors: list[str] = []
    if certificate.get("format") != "abi-capability-compiler-phase0-certificate/1":
        errors.append("unexpected certificate format")
    if certificate.get("status") != "PASS":
        errors.append("certificate status is not PASS")
    if certificate.get("historical_evidence_changed") is not False:
        errors.append("historical evidence must remain unchanged")
    if certificate.get("new_training_performed") is not False:
        errors.append("Phase 0 cannot include new training")

    commit = str(certificate.get("implementation_commit", ""))
    if len(commit) != 40:
        errors.append("implementation commit must be a full Git object ID")
    locks = certificate.get("implementation_locks")
    if not isinstance(locks, Mapping) or len(locks) < 6:
        errors.append("implementation locks are incomplete")
    else:
        for relative_path, expected_hash in locks.items():
            try:
                content = _git_show(repository_root, commit, str(relative_path))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            observed_hash = hashlib.sha256(content).hexdigest()
            if observed_hash != expected_hash:
                errors.append(
                    f"implementation hash mismatch for {relative_path}: "
                    f"{observed_hash} != {expected_hash}"
                )

    verification = certificate.get("verification")
    if not isinstance(verification, Mapping):
        errors.append("verification block is missing")
    else:
        suite = verification.get("implementation_commit_full_test_suite")
        if not isinstance(suite, Mapping) or int(suite.get("failed", -1)) != 0:
            errors.append("implementation commit full suite did not pass")
        if int(suite.get("passed", 0)) < 426:
            errors.append("implementation commit test depth is insufficient")
        certification_suite = verification.get("certification_tree_full_test_suite")
        if not isinstance(certification_suite, Mapping) or int(certification_suite.get("failed", -1)) != 0:
            errors.append("certification tree full suite did not pass")
        elif int(certification_suite.get("passed", 0)) < 431:
            errors.append("certification tree test depth is insufficient")
        if verification.get("phase0_protocol_verifier") != "PASS":
            errors.append("Phase 0 protocol verifier did not pass")
        if verification.get("root_json_parse_failures") != 0:
            errors.append("root JSON parse failures must be zero")

    transition = certificate.get("phase_transition")
    if not isinstance(transition, Mapping):
        errors.append("phase transition is missing")
    else:
        if transition.get("phase0") != "COMPLETE":
            errors.append("Phase 0 must transition to COMPLETE")
        if transition.get("phase1") != "OPEN":
            errors.append("Phase 1 must transition to OPEN")
        if transition.get("phase2_through_phase8") != "LOCKED":
            errors.append("later phases must remain locked")
    if certificate.get("moonshot_complete") is not False:
        errors.append("Phase 0 cannot complete the moonshot")
    return errors


def load_certificate(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("certificate root must be an object")
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path("ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    certificate_path = args.certificate
    if not certificate_path.is_absolute():
        certificate_path = args.repository_root / certificate_path
    errors = validate_certificate(
        load_certificate(certificate_path),
        args.repository_root,
    )
    result = {
        "format": "abi-capability-compiler-phase0-certificate-verification/1",
        "certificate": certificate_path.name,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
