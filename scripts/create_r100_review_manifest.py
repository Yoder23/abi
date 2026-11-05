"""Write-once public inventory for the completed capacity-control experiment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.acquisition_capacity_r100 import core as c  # noqa: E402
from experiments.grounded_english_r99 import fit_audit  # noqa: E402


def main():
    manifest = c.verify_binding()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for relative in manifest["files"]:
        if not relative.startswith("../layercake_release/") and relative not in tracked:
            raise RuntimeError(f"Unpublished required frozen input: {relative}")
    paths = set()
    for directory in (c.OUT, fit_audit.OUT):
        paths.update(
            p
            for p in directory.rglob("*")
            if p.is_file() and not p.name.startswith("trainable-step")
        )
    paths.update(c.checkpoint(arm) / "model.safetensors" for arm in c.ARMS)
    paths.update(
        ROOT / "results/review_2026_11_05" / name
        for name in (
            "r100-preflight-default.xml",
            "r100-preflight-phase7.xml",
            "r100-ci-surface.xml",
        )
    )
    inventory = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"Required evidence not staged/tracked: {relative}")
        inventory[relative] = {"sha256": c.sha(path), "bytes": path.stat().st_size}
    decision = c.read(c.OUT / "decision.json")
    if decision["verdict"] != "CAPACITY_CONTROL_COMPLETED":
        raise RuntimeError("Incomplete control")
    output = ROOT / "results/review_2026_11_05/r100_manifest.json"
    value = {
        "format": "r100-public-review-manifest/1",
        "intended_tag": "abi-research-r100-2026-11-05",
        "source_commit_at_inventory": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "numpy",
                "psutil",
                "huggingface_hub",
                "pytest",
                "ruff",
            )
        },
        "files": inventory,
        "required_input_manifest": {
            "path": "results/acquisition_capacity_r100/freeze.json",
            "sha256": c.sha(c.OUT / "freeze.json"),
        },
        "sibling_host_source_commit": c.HOST_COMMIT,
        "source_teacher": {
            "model": c.SOURCE,
            "revision": c.REVISION,
            "weights_in_repository": False,
        },
        "interpretation": "Conventional full-model capacity control; not an ABI product or novelty claim",
        "full_moonshot": "NOT_PROVEN",
        "human_fluency": "UNASSESSED",
        "independent_hardware": "NOT_EXECUTED",
        "intermediate_training_snapshots": "Preserved locally; final state and raw logs are published",
    }
    c.write_once(output, value)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": c.sha(output),
                "assets": len(inventory),
                "asset_bytes": sum(v["bytes"] for v in inventory.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
