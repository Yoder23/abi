import json
from pathlib import Path

import pytest

from abi.final_mile import FinalMileError
from abi.final_mile_release import build_release
from abi.reproduce import external_preflight, verify_portability, verify_release

ROOT = Path(__file__).resolve().parents[1]


def _attestation(path: Path):
    value = {
        "format": "abi-capability-compiler-phase8-external-operator-attestation/1",
        "operator_id": "external-test-operator",
        "independent_of_abi_development": True,
        "independent_hardware_owned_or_controlled_by_operator": True,
        "clean_abi_commit_verified": True,
        "clean_layercake_commit_verified": True,
        "release_manifest_verified_before_execution": True,
        "artifact_hashes_verified_before_execution": True,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_release_verification_passes_but_does_not_promote(tmp_path):
    release = tmp_path / "abi-release"
    build_release(ROOT, output=release, custody_key=tmp_path / "key.pem")
    result = verify_release(release)
    assert result["status"] == "PASS_RELEASE_BYTES_AND_OUTER_SIGNATURE"
    assert result["release_certified"] is False
    portability = verify_portability(release)
    assert portability["status"] == "HOST_INDEPENDENCE_FAILED"


def test_development_hardware_is_rejected_as_external(tmp_path):
    attestation = tmp_path / "attestation.json"
    _attestation(attestation)
    with pytest.raises(FinalMileError, match="development hardware"):
        external_preflight(attestation=attestation, require_cuda=False)


def test_tampered_release_file_is_rejected(tmp_path):
    release = tmp_path / "abi-release"
    build_release(ROOT, output=release, custody_key=tmp_path / "key.pem")
    target = release / "release-report.md"
    target.chmod(0o600)
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(FinalMileError, match="inventory changed"):
        verify_release(release)
