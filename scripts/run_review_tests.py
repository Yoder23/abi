"""Run research tests with an isolated, exact historical Phase 7 file fixture.

This is evidence recomputation, not live inference or independent hardware.
The historical verifier, tests, protocols, and sibling checkout are not edited.
Missing LFS payloads, Git objects, or hashes fail closed. No network fetch occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_capability_compiler_phase7_verify.py"
PROTOCOL = "ABI_CAPABILITY_COMPILER_PHASE7_ALLOCATION_BOUNDED_VERIFY_PROTOCOL_V1061.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(workspace: Path) -> Path:
    workspace = workspace.resolve()
    root = workspace / "abi_release"
    root.mkdir()
    document = json.loads((ROOT / PROTOCOL).read_text(encoding="utf-8"))
    base = json.loads((ROOT / document["base_protocol"]).read_text(encoding="utf-8"))
    bindings = {**base["bindings"], **document["bindings"]}
    files = set(bindings) | {PROTOCOL, document["base_protocol"], TEST}
    result_dir = (
        "results/abi_capability_compiler_phase7_integrated/allocation_bounded_verify_v1063/cpu/"
    )
    files |= {result_dir + "result.json", result_dir + "observations.jsonl"}
    commit = document["layercake_commit"]
    host = ROOT.parent / "layercake_release"
    for relative in sorted(files):
        target = (root / relative).resolve()
        if not target.is_relative_to(workspace):
            raise RuntimeError(f"Fixture path escapes isolated workspace: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith("../layercake_release/"):
            source_path = relative.removeprefix("../layercake_release/")
            payload = subprocess.check_output(
                ["git", "-C", str(host), "show", f"{commit}:{source_path}"]
            )
            target.write_bytes(payload)
        else:
            source = (ROOT / relative).resolve()
            if not source.is_relative_to(ROOT):
                raise RuntimeError(f"Unrecognized repository boundary: {relative}")
            shutil.copyfile(source, target)
        if relative in bindings and digest(target) != bindings[relative]:
            raise RuntimeError(f"Exact historical fixture hash mismatch: {relative}")
    print(f"Phase 7 fixture: {len(files)} exact files; LayerCake {commit}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-only", action="store_true")
    parser.add_argument("--junit-prefix", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    junit = {}
    if args.junit_prefix:
        prefix = args.junit_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for name in ("default", "phase7"):
            path = prefix.with_name(prefix.name + f"-{name}.xml")
            if path.exists():
                raise RuntimeError(f"Preserve prior test receipt: {path}")
            junit[name] = [f"--junitxml={path}"]
    if not args.phase7_only:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--ignore={TEST}", *junit.get("default", [])],
            cwd=ROOT,
            env=env,
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix="abi-phase7-exact-") as temporary:
        fixture = prepare(Path(temporary))
        # The original test derives its root from its own __file__. All its
        # reads therefore use the exact isolated fixture, not the current host.
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(fixture / TEST), *junit.get("phase7", [])],
            cwd=fixture,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
