"""Publish a write-once inventory for the completed no-training diagnostic."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.execution_continuation_r101 import core as c  # noqa: E402


def main():
    frozen = c.binding()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for relative in frozen["files"]:
        if not relative.startswith("../layercake_release/") and relative not in tracked:
            raise RuntimeError(f"Required input is not tracked: {relative}")
    if not (c.OUT / "decision.json").is_file() or (c.OUT / "stop.json").exists():
        raise RuntimeError("Diagnostic is incomplete or stopped")
    paths = {p for p in c.OUT.rglob("*") if p.is_file()}
    paths.update(c.old.checkpoint(a) / "model.safetensors" for a in c.ARMS)
    paths.update(ROOT / "results/review_2026_11_05" / n for n in
                 ("r101-default.xml", "r101-phase7.xml", "r101-ci-surface.xml"))
    inventory = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"Evidence not tracked: {relative}")
        inventory[relative] = {"sha256": c.sha(path), "bytes": path.stat().st_size}
    output = ROOT / "results/review_2026_11_05/r101_manifest.json"
    c.write_once(output, {
        "format": "r101-public-review-manifest/1", "intended_tag": "abi-research-r101-2026-11-05",
        "source_commit_at_inventory": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": inventory, "freeze_sha256": c.sha(c.OUT / "freeze.json"),
        "host_checkout": "865a2b9e59e6221caa1185f9b64479d2077953f3",
        "host_python_binding": c.old.HOST_COMMIT, "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("torch", "transformers", "tokenizers", "safetensors", "numpy", "psutil", "pytest")},
        "scope": "Execution-path and oracle-assisted continuation diagnosis; no training or teacher queries",
        "full_moonshot": "NOT_PROVEN", "independent_hardware": "NOT_EXECUTED",
    })
    print(json.dumps({"sha256": c.sha(output), "assets": len(inventory),
                      "bytes": sum(v["bytes"] for v in inventory.values())}))


if __name__ == "__main__":
    main()
