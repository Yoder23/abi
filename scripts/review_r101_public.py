"""Recompute R101 from pinned public checkouts, without live GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.execution_continuation_r101 import core as c  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve existing public-review receipt")
    receipt = {"format": "r101-public-review/1", "commands": [], "start_unix": time.time(),
               "scope": "Public-only sibling checkouts; same-machine evidence recomputation, not live generation or independent hardware",
               "full_moonshot": "NOT_PROVEN"}
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)

    def run(folder, command):
        done = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=900)
        receipt["commands"].append({"command": command, "cwd": str(folder), "returncode": done.returncode,
                                     "stdout": done.stdout, "stderr": done.stderr})
        print(json.dumps({"command": command, "returncode": done.returncode}), flush=True)
        if done.returncode:
            raise RuntimeError("Public review command failed")
        return done.stdout.strip()

    try:
        manifest_path = ROOT / "results/review_2026_11_05/r101_manifest.json"
        if c.sha(manifest_path) != args.manifest_sha256:
            raise RuntimeError("Public manifest changed")
        manifest = c.read(manifest_path)
        for folder, commit in ((ROOT, args.abi_commit), (c.HOST, manifest["host_checkout"])):
            if run(folder, ["git", "rev-parse", "HEAD"]) != commit:
                raise RuntimeError("Wrong public checkout")
            if run(folder, ["git", "status", "--porcelain=v1"]):
                raise RuntimeError("Dirty public checkout")
            run(folder, ["git", "remote", "get-url", "origin"])
        if not manifest["files"]:
            raise RuntimeError("Missing asset manifest")
        for relative, expected in manifest["files"].items():
            path = ROOT / relative
            if {"sha256": c.sha(path), "bytes": path.stat().st_size} != expected:
                raise RuntimeError(f"Missing or changed public asset: {relative}")
        receipt["assets"] = manifest["files"]
        receipt["manifest_sha256"] = c.sha(manifest_path)
        receipt["frozen_inputs"] = len(c.binding()["files"])
        run(ROOT, [sys.executable, "scripts/check_docs.py"])
        run(ROOT, [sys.executable, "-m", "experiments.execution_continuation_r101.run", "verify"])
        run(ROOT, [sys.executable, "-m", "pytest", "-q", "tests/test_execution_continuation_r101.py",
                   "tests/test_acquisition_capacity_r100.py", "tests/test_r99_fit_audit.py"])
        for folder in (ROOT, c.HOST):
            if run(folder, ["git", "status", "--porcelain=v1"]):
                raise RuntimeError("Public check mutated source")
        receipt["verdict"] = "BOUNDED_DIAGNOSTIC_EVIDENCE_RECOMPUTED"
    except Exception as error:
        receipt["verdict"] = "FAILED"
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["end_unix"] = time.time()
        c.write_once(args.output, receipt)
        print(json.dumps({"receipt_sha256": c.sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
