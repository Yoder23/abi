"""Integration checks for the real V1089 Phase 8 handoff inventory.

Unlike the frozen unit-level mutation tests, this module reads every entry in
the actual 52-file manifest. ABI payloads must be present in this checkout and
LayerCake source bytes must exist in the exact bound Git commit. A missing LFS
payload, missing sibling repository, missing commit, or mismatched byte fails
closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from abi.capability_compiler_phase2_common import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "results/abi_capability_compiler_phase8/readiness_v1073/manifest.json"
)
LAYERCAKE_ROOT = (ROOT / "../layercake_release").resolve()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={LAYERCAKE_ROOT.as_posix()}",
            "-C",
            str(LAYERCAKE_ROOT),
            *args,
        ],
        check=False,
        capture_output=True,
    )


def _manifest() -> dict:
    assert MANIFEST_PATH.is_file(), "V1089 Phase 8 manifest is absent"
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert document["format"] == (
        "abi-capability-compiler-phase8-release-readiness-result/1"
    )
    assert document["file_count"] == 52
    assert len(document["files"]) == 52
    assert document["total_bytes"] == 281_108_851
    return document


def test_real_manifest_evidence_hash_and_external_claim_ceiling() -> None:
    document = _manifest()
    declared = document.pop("evidence_sha256")
    assert declared == _sha256_bytes(canonical_json_bytes(document))
    assert document["phase8_certified"] is False
    assert not any(document["external_gates"].values())


def test_every_real_manifest_byte_is_available_and_exact() -> None:
    document = _manifest()
    layercake_commit = document["source"]["layercake_commit"]
    assert LAYERCAKE_ROOT.is_dir(), "required sibling LayerCake checkout is absent"
    commit_check = _git("cat-file", "-e", f"{layercake_commit}^{{commit}}")
    assert commit_check.returncode == 0, (
        "bound LayerCake commit is absent from sibling repository: "
        f"{layercake_commit}"
    )

    recomputed_bytes = 0
    for relative, expected in sorted(document["files"].items()):
        if expected["repository"] == "abi":
            target = ROOT / relative
            assert target.is_file(), relative
            assert target.stat().st_size == expected["bytes"], relative
            assert _sha256_file(target) == expected["sha256"], relative
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            assert tracked.returncode == 0, f"public ABI path is untracked: {relative}"
        else:
            prefix = "../layercake_release/"
            assert relative.startswith(prefix), relative
            repository_path = relative[len(prefix) :]
            shown = _git("show", f"{layercake_commit}:{repository_path}")
            assert shown.returncode == 0, relative
            assert len(shown.stdout) == expected["bytes"], relative
            assert _sha256_bytes(shown.stdout) == expected["sha256"], relative
        recomputed_bytes += expected["bytes"]

    assert recomputed_bytes == document["total_bytes"]
