import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from abi.final_mile import sha256_file
from abi.final_mile_release import EXPECTED_PACKAGE_HASHES, build_release

ROOT = Path(__file__).resolve().parents[1]


def test_failed_candidate_release_is_complete_signed_and_non_promotional(tmp_path):
    output = tmp_path / "abi-release"
    result = build_release(ROOT, output=output, custody_key=tmp_path / "release-key.pem")
    assert result["certificate"]["status"] == "HOST_INDEPENDENCE_FAILED"
    assert result["certificate"]["release_certified"] is False
    for name, expected in EXPECTED_PACKAGE_HASHES.items():
        assert sha256_file(output / name) == expected
    required = {
        "canonical_host_abi.json",
        "receiver-certification-spec.json",
        "imported-information-ledger.json",
        "source-success-lock.json",
        "compatibility-matrix.json",
        "release-certificate.json",
        "release-report.md",
        "release-manifest.json",
        "release-signature.json",
    }
    assert required <= {path.name for path in output.rglob("*") if path.is_file()}

    signature = json.loads((output / "release-signature.json").read_text(encoding="utf-8"))
    public = serialization.load_pem_public_key(signature["public_key_pem"].encode())
    assert isinstance(public, Ed25519PublicKey)
    public.verify(
        bytes.fromhex(signature["signature_ed25519_hex"]),
        (output / "release-manifest.json").read_bytes(),
    )


def test_source_success_lock_has_exact_frozen_successes(tmp_path):
    output = tmp_path / "abi-release"
    build_release(ROOT, output=output, custody_key=tmp_path / "release-key.pem")
    lock = json.loads((output / "source-success-lock.json").read_text(encoding="utf-8"))
    assert lock["successful_tasks"] == 1381
    assert len(set(lock["successful_task_ids"])) == 1381
    assert lock["required_receiver_retention"] == 1.0
