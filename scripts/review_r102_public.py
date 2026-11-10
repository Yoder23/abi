"""Recompute R102 from pinned public siblings; no training or live generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, files):
    root = root.resolve()
    if not files:
        raise RuntimeError("Empty review inventory")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Missing or external asset: {relative}")
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise RuntimeError(f"Changed public asset: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi-root", type=Path, required=True)
    parser.add_argument("--abi-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.abi_root.resolve(), args.output.resolve()
    host = root.parent / "layercake_release"
    if output.exists() or any(output.is_relative_to(p) for p in (root, host)):
        raise RuntimeError("Use a new receipt path outside both checkouts")
    receipt = {"format": "r102-public-review/1", "commands": [], "start_unix": time.time(),
               "helper_sha256": sha(Path(__file__)), "full_moonshot": "NOT_PROVEN",
               "scope": "Pinned public sibling checkouts; same-machine evidence recomputation, not live generation, human review or independent hardware"}
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    os.environ.update({k: env[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})
    sys.path.insert(0, str(root))

    def run(folder, command):
        done = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=900)
        receipt["commands"].append({"command": command, "cwd": str(folder), "returncode": done.returncode,
                                     "stdout": done.stdout, "stderr": done.stderr})
        print(json.dumps({"command": command, "returncode": done.returncode}), flush=True)
        if done.returncode:
            raise RuntimeError("Public review command failed")
        return done.stdout.strip()

    try:
        manifest_path = root / "results/review_2026_11_10/r102_manifest.json"
        if sha(manifest_path) != args.manifest_sha256:
            raise RuntimeError("Public manifest identity mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for folder, commit, remote in (
            (root, args.abi_commit, "https://github.com/Yoder23/abi.git"),
            (host, manifest["sibling_host_commit"], "https://github.com/Yoder23/layercake.git"),
        ):
            if run(folder, ["git", "rev-parse", "HEAD"]) != commit:
                raise RuntimeError("Wrong public checkout")
            if run(folder, ["git", "remote", "get-url", "origin"]) != remote:
                raise RuntimeError("Unexpected source remote")
            if run(folder, ["git", "status", "--porcelain=v1"]):
                raise RuntimeError("Dirty public checkout")
        verify_files(root, manifest["files"])
        from experiments.coverage_exposure_r102 import core as c

        if c.ROOT.resolve() != root or c.HOST.resolve() != host:
            raise RuntimeError("Imported a development checkout")
        receipt["manifest_sha256"] = sha(manifest_path)
        receipt["assets"] = len(manifest["files"])
        receipt["lfs_paths"] = len(manifest["lfs_paths"])
        receipt["frozen_inputs"] = len(c.binding()["files"])
        run(root, [sys.executable, "scripts/check_docs.py"])
        run(root, [sys.executable, "-m", "experiments.coverage_exposure_r102.run", "verify"])
        run(root, [sys.executable, "-m", "pytest", "-q", "tests/test_coverage_exposure_r102.py",
                   "tests/test_r102_error_audit.py", "tests/test_acquisition_capacity_r100.py",
                   "tests/test_r99_fit_audit.py"])
        for folder in (root, host):
            if run(folder, ["git", "status", "--porcelain=v1"]):
                raise RuntimeError("Review mutated source")
        receipt["decision_sha256"] = sha(c.OUT / "decision.json")
        if receipt["decision_sha256"] != manifest["decision_sha256"]:
            raise RuntimeError("Scientific decision changed")
        receipt["verdict"] = "BOUNDED_NEGATIVE_FACTORIAL_EVIDENCE_RECOMPUTED"
    except Exception as error:
        receipt["verdict"] = "FAILED"
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["end_unix"] = time.time()
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        print(json.dumps({"receipt_sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
