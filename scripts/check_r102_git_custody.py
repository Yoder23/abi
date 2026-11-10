"""Check pinned local Git/LFS objects; deliberately not a public-clone claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.coverage_exposure_r102 import core as c  # noqa: E402


def blob(folder, revision, path):
    return subprocess.check_output(["git", "cat-file", "blob", f"{revision}:{path}"], cwd=folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve previous custody receipt")
    manifest_path = ROOT / "results/review_2026_11_10/r102_manifest.json"
    manifest = c.read(manifest_path)
    if blob(ROOT, args.revision, manifest_path.relative_to(ROOT).as_posix()) != manifest_path.read_bytes():
        raise RuntimeError("Manifest is not the pinned Git object")
    git_dir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True).strip())
    counts = {"git_files": 0, "lfs_paths": 0, "host_source_files": 0}
    for relative, expected in manifest["files"].items():
        data = blob(ROOT, args.revision, relative)
        if expected["storage"] == "git_lfs":
            pointer = data.decode("ascii").splitlines()
            if pointer != ["version https://git-lfs.github.com/spec/v1", f"oid sha256:{expected['sha256']}", f"size {expected['bytes']}"]:
                raise RuntimeError(f"Incorrect LFS pointer: {relative}")
            digest = expected["sha256"]
            payload = git_dir / "lfs/objects" / digest[:2] / digest[2:4] / digest
            if payload.stat().st_size != expected["bytes"] or c.sha(payload) != digest:
                raise RuntimeError(f"Missing local LFS object: {relative}")
            counts["lfs_paths"] += 1
        else:
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Git bytes differ from evidence: {relative}")
            counts["git_files"] += 1
    for relative, digest in manifest["external_host_inputs"].items():
        if hashlib.sha256(blob(c.HOST, manifest["host_source_binding_commit"], relative)).hexdigest() != digest:
            raise RuntimeError(f"Host Git source mismatch: {relative}")
        counts["host_source_files"] += 1
    c.binding()
    value = {"format": "r102-local-git-custody/1", "revision": args.revision,
             "manifest_sha256": c.sha(manifest_path), "counts": counts,
             "verdict": "PINNED_LOCAL_GIT_AND_LFS_OBJECTS_PRESENT",
             "scope": "Local object custody only; no network, public reconstruction, fresh inference or independent hardware",
             "full_moonshot": "NOT_PROVEN"}
    c.write_once(args.output, value)
    print(json.dumps({**value, "receipt_sha256": c.sha(args.output)}))


if __name__ == "__main__":
    main()
