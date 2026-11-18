"""Publish completed R111 evidence after frozen raw recomputation, without inference.

This additive custody helper does not change the experiment or supply a new
scientific verdict. Historical Git/LFS, host and source-cache dependencies remain
required. Neither this index nor its archive is independent reproduction.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.package_r110_evidence import sha, write_new  # noqa: E402

OUT = ROOT / "results/review_2026_11_18"
MANIFEST = OUT / "r111_manifest.json"
RECEIPT = OUT / "r111_strict_recomputation.json"
ARCHIVE = OUT / "r111_evidence.zip"
ARCHIVE_RECEIPT = OUT / "r111_archive.json"


def checked_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise RuntimeError(f"Missing or escaping publication path: {name}")
    return path


def package():
    if any(path.exists() for path in (MANIFEST, RECEIPT)):
        raise RuntimeError("Preserve the existing publication packet; do not overwrite")
    from experiments.scope_revision_r111 import design as d
    from experiments.scope_revision_r111 import run as r

    started, transcript = time.monotonic(), io.StringIO()
    print("Recomputing frozen R111 evidence; no model calls", flush=True)
    with contextlib.redirect_stdout(transcript):
        result = r.verify()
    elapsed = time.monotonic() - started
    frozen = r.read(d.OUT / "freeze.json")
    scope_failed = result["verdict"] == "SCOPE_PREREQUISITE_FAILED"
    result_path = "results/scope_revision_r111/" + (
        "scope/decision.json" if scope_failed else "live/decision.json")
    stored = r.read(checked_path(result_path))
    if (not scope_failed and stored != result) or (
            scope_failed and any(result.get(k) != v for k, v in stored.items())):
        raise RuntimeError("Recomputed result differs from the immutable decision")
    names = {name for name in frozen["files"] if not name.startswith("../")}
    names.update(p.relative_to(ROOT).as_posix() for p in d.OUT.rglob("*") if p.is_file())
    names.update(("scripts/package_r111_evidence.py", "tests/test_r111_publication.py",
                  "docs/R111_SCOPE_REVISION.md", "docs/releases/R111_REVIEW.md"))
    names.update(p.relative_to(ROOT).as_posix() for p in OUT.glob("r111-*.xml"))
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    if names - tracked:
        raise RuntimeError(f"Stage required files first: {sorted(names - tracked)}")
    for name in names:
        checked_path(name)
    lfs = set(subprocess.check_output(["git", "lfs", "ls-files", "-n"],
                                      cwd=ROOT, text=True).splitlines()) & names
    write_new(RECEIPT, {
        "format": "r111-strict-recomputation/1",
        "verification_entrypoint": "experiments.scope_revision_r111.run.verify",
        "elapsed_seconds": elapsed,
        "result_path": result_path,
        "result_sha256": sha(ROOT / result_path),
        "stdout": transcript.getvalue(),
        "model_calls": 0,
        "scope": "Same-machine raw recomputation, not new live or independent-hardware execution",
    })
    names.add(RECEIPT.relative_to(ROOT).as_posix())
    manifest = {
        "format": "r111-public-manifest/1",
        "tag": "abi-research-r111-2026-11-18",
        "files": {name: {"sha256": sha(checked_path(name)),
                         "bytes": checked_path(name).stat().st_size} for name in sorted(names)},
        "host_inputs": {name: digest for name, digest in frozen["files"].items()
                        if name.startswith("../")},
        "host_revision": r.old.h.HOST_REVISION,
        "source_model": r.old.h.SOURCE,
        "source_revision": r.old.h.REVISION,
        "source_files": frozen["source_files"],
        "packages": frozen["packages"],
        "lfs_paths": sorted(lfs),
        "result_path": result_path,
        "result_sha256": sha(ROOT / result_path),
        "freeze_sha256": sha(d.OUT / "freeze.json"),
        "scope_decision_sha256": sha(d.OUT / "scope/decision.json"),
        "scope": "Custody index; historical Git/LFS, host and pinned source cache remain required",
    }
    write_new(MANIFEST, manifest)
    print(json.dumps({"manifest_sha256": sha(MANIFEST), "assets": len(names),
                      "host_bindings": len(manifest["host_inputs"]),
                      "strict_recomputation_seconds": elapsed,
                      "verdict": result["verdict"]}), flush=True)


def archive():
    """Archive only indexed new evidence; check every member against its manifest."""
    if ARCHIVE.exists() or ARCHIVE_RECEIPT.exists():
        raise RuntimeError("Preserve the existing archive packet; do not overwrite")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = sorted(name for name in manifest["files"]
                   if name.startswith("results/scope_revision_r111/"))
    if not names:
        raise RuntimeError("Missing R111 archive inventory")
    for name in names:
        path = checked_path(name)
        item = manifest["files"][name]
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Changed archive input: {name}")
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(checked_path(name), name)
    import hashlib
    with zipfile.ZipFile(ARCHIVE) as bundle:
        if bundle.namelist() != names:
            raise RuntimeError("Archive membership mismatch")
        for name in names:
            if hashlib.sha256(bundle.read(name)).hexdigest() != manifest["files"][name]["sha256"]:
                raise RuntimeError(f"Archive member mismatch: {name}")
    write_new(ARCHIVE_RECEIPT, {
        "format": "r111-new-evidence-archive/1", "archive": ARCHIVE.name,
        "sha256": sha(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "members": names,
        "manifest_sha256": sha(MANIFEST),
        "scope": "New R111 evidence only; not a self-contained dependency/source-model archive",
    })
    print(json.dumps({"archive_sha256": sha(ARCHIVE), "members": len(names),
                      "bytes": ARCHIVE.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "archive"))
    args = parser.parse_args()
    {"package": package, "archive": archive}[args.command]()
