"""Recompute the R104 prerequisite result from pinned public sibling checkouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


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
        raise RuntimeError("Preserve previous receipts; output must be outside both checkouts")
    sys.path.insert(0, str(root))
    from scripts.review_r102_public import sha, verify_files

    receipt = {"format": "r104-public-review/1", "commands": [], "start_unix": time.time(),
        "helper_sha256": sha(Path(__file__)), "full_moonshot": "NOT_PROVEN",
        "scope": "Same-machine public evidence recomputation, not live generation, human review or independent hardware"}
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    os.environ.update({k: env[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})

    def run(command, folder=root):
        done = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=1800)
        receipt["commands"].append({"command": command, "cwd": str(folder), "returncode": done.returncode,
                                     "stdout": done.stdout, "stderr": done.stderr})
        print(json.dumps({"command": command, "returncode": done.returncode}), flush=True)
        if done.returncode:
            raise RuntimeError("Public review command failed")
        return done.stdout.strip()

    try:
        manifest_path = root / "results/review_2026_11_10/r104_manifest.json"
        if sha(manifest_path) != args.manifest_sha256:
            raise RuntimeError("Public manifest mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha(Path(__file__)) != manifest["files"]["scripts/review_r104_public.py"]["sha256"]:
            raise RuntimeError("Review helper differs from published inventory")
        for folder, commit, remote in ((root, args.abi_commit, "https://github.com/Yoder23/abi.git"),
                                      (host, manifest["sibling_host_commit"], "https://github.com/Yoder23/layercake.git")):
            if run(["git", "rev-parse", "HEAD"], folder) != commit:
                raise RuntimeError("Wrong pinned revision")
            if run(["git", "remote", "get-url", "origin"], folder) != remote or run(["git", "status", "--porcelain=v1"], folder):
                raise RuntimeError("Wrong remote or dirty source checkout")
        verify_files(root, manifest["files"])
        from experiments.evaluator_control_r104 import core as c

        if c.ROOT.resolve() != root or c.HOST.resolve() != host:
            raise RuntimeError("Imported development code")
        receipt.update(assets=len(manifest["files"]), lfs_paths=len(manifest["lfs_paths"]),
                       frozen_inputs=len(c.binding()["files"]), manifest_sha256=sha(manifest_path))
        run([sys.executable, "scripts/check_docs.py"])
        run([sys.executable, "-m", "experiments.evaluator_control_r104.run", "verify"])
        run([sys.executable, "-m", "experiments.evaluator_control_r104.error_audit"])
        run([sys.executable, "-m", "pytest", "-q", "tests/test_evaluator_control_r104.py", "tests/test_r104_completed_verifier.py"])
        for folder in (root, host):
            if run(["git", "status", "--porcelain=v1"], folder):
                raise RuntimeError("Review mutated source")
        receipt["decision_sha256"] = sha(c.OUT / "decision.json")
        if receipt["decision_sha256"] != manifest["decision_sha256"]:
            raise RuntimeError("Changed scientific decision")
        receipt["verdict"] = "BOUNDED_R104_DECISION_RECOMPUTED"
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
